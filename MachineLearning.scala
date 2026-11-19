import java.nio.file.{Files, Paths}

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.NumericType
import org.apache.spark.ml.{Pipeline, PipelineStage, Transformer}
import org.apache.spark.ml.attribute.AttributeGroup
import org.apache.spark.ml.classification.{
  DecisionTreeClassificationModel, DecisionTreeClassifier,
  LogisticRegression, LogisticRegressionModel,
  RandomForestClassificationModel, RandomForestClassifier
}
import org.apache.spark.ml.evaluation.BinaryClassificationEvaluator
import org.apache.spark.ml.feature.{OneHotEncoder, StandardScaler, StringIndexer, VectorAssembler}
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.mllib.evaluation.MulticlassMetrics

// ==========================================================================
// Phase 5 - Machine Learning (RT-IoT2022)
//
// Problem : multiclass classification of a network flow into one of the 12
//           Attack_type classes (3 normal IoT classes + 9 attack classes).
//           A binary view (normal vs attack) is derived from the same
//           predictions and reported as a second, security-oriented result.
// Input   : data/preprocessed_dataset.parquet
//           (Phase 2: duplicates removed, constant/correlated features
//            dropped, engineered features, log1p on skewed features,
//            NOT scaled, Attack_type already indexed into "label")
// Output  : results/ml_metrics.txt and results/ml_*.csv (+ saved models)
//
// Leakage rules applied in this file
//   1. The data is split FIRST (stratified 70/30, fixed seed).
//   2. StringIndexer, OneHotEncoder and StandardScaler are fitted on the
//      TRAINING split only and then applied to the test split.
//   3. attack_family, traffic_class, is_attack and Attack_type are derived
//      from the target. They are never model inputs.
//      (traffic_class / is_attack are only used AFTER prediction, to score
//       the binary normal-vs-attack view.)
//   4. Class weights are computed from the training split only.
//   5. No model or hyper-parameter is chosen on the test split.
// ==========================================================================

object MachineLearning {

  // ---------------- configuration ----------------
  val Seed: Long            = 42L
  val TrainFraction: Double = 0.7
  val WeightPower: Double   = 1.0   // 1.0 = fully balanced class weights, 0.5 = square-root weights, 0.0 = none
  val NumTrees: Int         = 100
  val TreeDepth: Int        = 12

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("RT-IoT2022 Machine Learning")
      .master("local[*]")
      .config("spark.sql.shuffle.partitions", "16")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    import spark.implicits._

    // Everything printed is also kept so it can be saved as results/ml_metrics.txt
    val log = new StringBuilder
    def out(s: String = ""): Unit = { println(s); log.append(s).append("\n") }
    def header(title: String): Unit = {
      out("=" * 70); out(title); out("=" * 70)
    }
    new java.io.File("results").mkdirs()
    def writeFile(path: String, lines: Seq[String]): Unit =
      Files.write(Paths.get(path), lines.mkString("\n").getBytes("UTF-8"))

    // ======================================================================
    // 0. Load, choose the columns, split
    // ======================================================================
    header("0. Load the preprocessed dataset and build the feature list")

    val raw = spark.read.parquet("data/preprocessed_dataset.parquet")
    require(raw.columns.contains("label"), "Column 'label' (indexed Attack_type) not found - run Phase 2 first")

    // Columns that describe or are derived from the target: never inputs
    val targetDerived = Set("Attack_type", "attack_family", "traffic_class", "is_attack", "label")

    val catCols = Array("proto", "service", "orig_port_type", "resp_port_type")
      .filter(raw.columns.contains)

    // Numeric inputs = every numeric column that is not target-derived.
    // (The old *_vec one-hot columns from Phase 2 are vectors, so they are
    //  ignored here and the encoders are refitted on the training split.)
    val numCols = raw.schema.fields
      .filter(f => f.dataType.isInstanceOf[NumericType] && !targetDerived.contains(f.name))
      .map(_.name)

    // Some Phase 2 columns contain dots (e.g. "fwd_iat.max"); VectorAssembler
    // reads dots as nested fields, so inputs get safe names and are mapped back.
    val safeNames = numCols.indices.map(i => s"num_$i").toArray
    val safeToOriginal: Map[String, String] = safeNames.zip(numCols).toMap
    val originalToSafe: Map[String, String] = numCols.zip(safeNames).toMap

    val selectExprs =
      Seq(col("label").cast("double").alias("label"),
          col("Attack_type"),
          col("traffic_class"),
          col("is_attack").cast("double").alias("is_attack")) ++
      catCols.map(c => col(s"`$c`")) ++
      numCols.zip(safeNames).map { case (o, s) => col(s"`$o`").cast("double").alias(s) }

    val data = raw.select(selectExprs: _*).na.fill(0.0, safeNames)

    val total = data.count()
    out("Rows:                  " + total)
    out("Numeric input columns: " + numCols.length)
    out("Categorical inputs:    " + catCols.mkString(", "))
    out("Excluded (derived from the target): " + targetDerived.toSeq.sorted.mkString(", "))

    val labelNames: Map[Double, String] =
      data.select("label", "Attack_type").distinct().collect()
        .map(r => (r.getDouble(0), r.getString(1))).toMap
    val allLabels: Array[Double] = labelNames.keys.toArray.sorted
    val numClasses = allLabels.length

    val normalLabels: Array[Double] =
      data.filter(col("traffic_class") === "Normal").select("label").distinct()
        .collect().map(_.getDouble(0)).sorted
    out("Classes: " + numClasses + "  (normal classes: " +
        normalLabels.map(l => labelNames(l)).mkString(", ") + ")")

    // ---- Stratified 70/30 split -------------------------------------------
    // Inside each class the rows are ordered by a hash of the row itself, and
    // the first 70% go to training. The hash does not depend on how Spark
    // partitions the data, so the split is identical on every run and machine.
    // Every class (including the 28-flow FIN scan) appears in both splits.
    header("1. Stratified train / test split (70/30, seed " + Seed + ")")

    val hashCols = (safeNames.toSeq ++ catCols.toSeq ++ Seq("label")).map(c => col(c))
    val ranked = data
      .withColumn("_h", xxhash64(hashCols: _*))
      .withColumn("_rn", row_number().over(Window.partitionBy("label").orderBy("_h")))
      .withColumn("_cnt", count(lit(1)).over(Window.partitionBy("label")))
      .cache()
    ranked.count()

    val cutoff = round(col("_cnt") * TrainFraction)
    val trainDf = ranked.filter(col("_rn") <= cutoff).drop("_h", "_rn", "_cnt").cache()
    val testDf  = ranked.filter(col("_rn") >  cutoff).drop("_h", "_rn", "_cnt").cache()

    val nTrain = trainDf.count()
    val nTest  = testDf.count()
    out(f"Training rows: $nTrain%d (${100.0 * nTrain / total}%.1f%%)   Test rows: $nTest%d (${100.0 * nTest / total}%.1f%%)")
    require(nTrain + nTest == total, "split lost or duplicated rows")

    val trainCounts: Map[Double, Long] =
      trainDf.groupBy("label").count().collect().map(r => (r.getDouble(0), r.getLong(1))).toMap
    val testCounts: Map[Double, Long] =
      testDf.groupBy("label").count().collect().map(r => (r.getDouble(0), r.getLong(1))).toMap

    out(f"${"class"}%-28s ${"train"}%8s ${"test"}%8s ${"test %"}%8s")
    allLabels.sortBy(l => -(trainCounts.getOrElse(l, 0L) + testCounts.getOrElse(l, 0L))).foreach { l =>
      val tr = trainCounts.getOrElse(l, 0L); val te = testCounts.getOrElse(l, 0L)
      out(f"${labelNames(l)}%-28s $tr%8d $te%8d ${100.0 * te / (tr + te)}%8.1f")
    }
    require(allLabels.forall(l => trainCounts.contains(l) && testCounts.contains(l)),
            "a class is missing from the train or the test split")

    // ---- Class weights (training split only) ------------------------------
    // weight = (n / (k * n_class)) ^ WeightPower. With power 1 every class
    // carries the same total weight, so a 28-flow class counts as much as
    // the 90,000-flow SYN flood.
    val classWeights: Map[Double, Double] = trainCounts.map { case (l, c) =>
      l -> math.pow(nTrain.toDouble / (trainCounts.size * c), WeightPower)
    }
    val weightUdf = udf((l: Double) => classWeights(l))
    val trainW = trainDf.withColumn("weight", weightUdf(col("label")))

    // ======================================================================
    // 2. Feature pipeline - fitted on the TRAINING split only
    // ======================================================================
    header("2. Feature preparation (fitted on the training split only)")

    val indexers: Array[PipelineStage] = catCols.map { c =>
      new StringIndexer().setInputCol(c).setOutputCol(c + "_idx").setHandleInvalid("keep")
    }
    val encoder = new OneHotEncoder()
      .setInputCols(catCols.map(_ + "_idx"))
      .setOutputCols(catCols.map(_ + "_ohe"))
      .setHandleInvalid("keep")
    val assembler = new VectorAssembler()
      .setInputCols(safeNames ++ catCols.map(_ + "_ohe"))
      .setOutputCol("features_raw")
      .setHandleInvalid("keep")
    // withMean = false keeps the one-hot vectors sparse; the numeric inputs
    // are already log-compressed by Phase 2, so only the spread is scaled.
    val scaler = new StandardScaler()
      .setInputCol("features_raw").setOutputCol("features")
      .setWithStd(true).setWithMean(false)

    val prepModel = new Pipeline()
      .setStages(indexers ++ Array[PipelineStage](encoder, assembler, scaler))
      .fit(trainW)

    val trainP = prepModel.transform(trainW).cache()
    val testP  = prepModel.transform(testDf).cache()

    val attrGroup = AttributeGroup.fromStructField(trainP.schema("features_raw"))
    val rawNames: Array[String] =
      attrGroup.attributes.map(_.map(_.name.getOrElse("?"))).getOrElse(Array.tabulate(attrGroup.size)(i => s"f$i"))
    val featureNames: Array[String] = rawNames.map(n => safeToOriginal.getOrElse(n, n))
    out("Feature vector length: " + featureNames.length +
        " (" + numCols.length + " numeric + " + (featureNames.length - numCols.length) + " one-hot slots)")

    // Feature group of a vector slot: one-hot slots collapse to their column
    def groupOf(name: String): String =
      catCols.find(c => name.startsWith(c + "_ohe_")).getOrElse(name)

    // ======================================================================
    // Evaluation helpers
    // ======================================================================
    def mcMetrics(preds: DataFrame): MulticlassMetrics =
      new MulticlassMetrics(
        preds.select(col("prediction").cast("double"), col("label").cast("double"))
          .rdd.map(r => (r.getDouble(0), r.getDouble(1))))

    def macroF1(m: MulticlassMetrics, labels: Array[Double]): Double =
      labels.map(l => m.fMeasure(l)).sum / labels.length

    val pAttackUdf = {
      val idx = normalLabels.map(_.toInt)
      udf((p: Vector) => math.max(0.0, math.min(1.0, 1.0 - idx.map(i => p(i)).sum)))
    }

    case class MultiEval(name: String, trainAcc: Double, trainWF1: Double, acc: Double,
                         wPrec: Double, wRec: Double, wF1: Double, macroF: Double, fitSec: Double)
    case class BinEval(name: String, precision: Double, recall: Double, f1: Double,
                       falseAlarm: Double, miss: Double, auc: Option[Double], aucPr: Option[Double])

    def binaryEval(name: String, preds: DataFrame, hasProb: Boolean): BinEval = {
      val withPred = preds.withColumn("pred_attack",
        when(col("prediction").isin(normalLabels: _*), 0.0).otherwise(1.0))
      val c = withPred.groupBy("is_attack", "pred_attack").count().collect()
        .map(r => ((r.getDouble(0), r.getDouble(1)), r.getLong(2))).toMap
      def n(a: Double, p: Double): Double = c.getOrElse((a, p), 0L).toDouble
      val tp = n(1.0, 1.0); val fp = n(0.0, 1.0); val fn = n(1.0, 0.0); val tn = n(0.0, 0.0)
      val prec = if (tp + fp > 0) tp / (tp + fp) else 0.0
      val rec  = if (tp + fn > 0) tp / (tp + fn) else 0.0
      val f1   = if (prec + rec > 0) 2 * prec * rec / (prec + rec) else 0.0
      val (roc, pr) =
        if (hasProb) {
          val sc = preds.withColumn("p_attack", pAttackUdf(col("probability")))
          val ev = new BinaryClassificationEvaluator().setLabelCol("is_attack").setRawPredictionCol("p_attack")
          (Some(ev.setMetricName("areaUnderROC").evaluate(sc)), Some(ev.setMetricName("areaUnderPR").evaluate(sc)))
        } else (None, None)
      BinEval(name, prec, rec, f1, fp / math.max(1.0, fp + tn), fn / math.max(1.0, fn + tp), roc, pr)
    }

    // ======================================================================
    // 3. Models
    // ======================================================================
    header("3. Training and evaluation")

    val results = scala.collection.mutable.ArrayBuffer.empty[(MultiEval, BinEval, DataFrame)]

    def record(name: String, trainPreds: DataFrame, testPreds: DataFrame, fitSec: Double, hasProb: Boolean): Unit = {
      val mTrain = mcMetrics(trainPreds)
      val mTest  = mcMetrics(testPreds)
      val testLabels = testCounts.keys.toArray.sorted
      val me = MultiEval(name, mTrain.accuracy, mTrain.weightedFMeasure, mTest.accuracy,
        mTest.weightedPrecision, mTest.weightedRecall, mTest.weightedFMeasure,
        macroF1(mTest, testLabels), fitSec)
      val be = binaryEval(name, testPreds, hasProb)
      results += ((me, be, testPreds))
      out(f"  done: $name%-26s test accuracy = ${me.acc}%.4f   macro-F1 = ${me.macroF}%.4f   (fit ${me.fitSec}%.1f s)")
    }

    def timed[T](f: => T): (T, Double) = {
      val t0 = System.nanoTime(); val r = f; (r, (System.nanoTime() - t0) / 1e9)
    }

    // --- Baseline: always predict the most frequent training class ---------
    val majority = trainCounts.maxBy(_._2)._1
    out("Baseline: always predict '" + labelNames(majority) + "'")
    record("Majority-class baseline",
      trainP.withColumn("prediction", lit(majority)),
      testP.withColumn("prediction", lit(majority)), 0.0, hasProb = false)

    // --- Logistic regression (linear baseline, multinomial) ----------------
    val lr = new LogisticRegression()
      .setLabelCol("label").setFeaturesCol("features").setWeightCol("weight")
      .setFamily("multinomial").setMaxIter(100).setRegParam(1e-4).setStandardization(false)
    val (lrModel, lrSec) = timed(lr.fit(trainP))
    record("Logistic Regression (w)", lrModel.transform(trainP), lrModel.transform(testP), lrSec, hasProb = true)

    // --- Decision tree ----------------------------------------------------
    val dt = new DecisionTreeClassifier()
      .setLabelCol("label").setFeaturesCol("features").setWeightCol("weight")
      .setMaxDepth(TreeDepth).setSeed(Seed)
    val (dtModel, dtSec) = timed(dt.fit(trainP))
    record("Decision Tree (w)", dtModel.transform(trainP), dtModel.transform(testP), dtSec, hasProb = true)

    // --- Random forest, with and without class weights --------------------
    val rfBase = new RandomForestClassifier()
      .setLabelCol("label").setFeaturesCol("features")
      .setNumTrees(NumTrees).setMaxDepth(TreeDepth).setSeed(Seed)
    val (rfUModel, rfUSec) = timed(rfBase.fit(trainP))
    record("Random Forest (unweighted)", rfUModel.transform(trainP), rfUModel.transform(testP), rfUSec, hasProb = true)

    val (rfModel, rfSec) = timed(rfBase.copy(org.apache.spark.ml.param.ParamMap.empty).setWeightCol("weight").fit(trainP))
    val rfTrainPreds = rfModel.transform(trainP)
    val rfTestPreds  = rfModel.transform(testP).cache()
    record("Random Forest (w)", rfTrainPreds, rfTestPreds, rfSec, hasProb = true)

    // The random forest with class weights is the model chosen BEFORE seeing
    // the test results (trees + imbalance + mixed features, see the report).
    val primaryName = "Random Forest (w)"
    val primary = results.find(_._1.name == primaryName).get

    // ======================================================================
    // 4. Results tables
    // ======================================================================
    header("4. Model comparison - multiclass (12 classes)")
    out(f"${"model"}%-28s ${"tr.acc"}%7s ${"tr.wF1"}%7s | ${"acc"}%7s ${"w.prec"}%7s ${"w.rec"}%7s ${"w.F1"}%7s ${"macroF1"}%8s")
    results.foreach { case (m, _, _) =>
      out(f"${m.name}%-28s ${m.trainAcc}%7.4f ${m.trainWF1}%7.4f | ${m.acc}%7.4f ${m.wPrec}%7.4f ${m.wRec}%7.4f ${m.wF1}%7.4f ${m.macroF}%8.4f")
    }

    header("5. Model comparison - binary view (attack = positive class)")
    out(f"${"model"}%-28s ${"prec"}%7s ${"recall"}%7s ${"F1"}%7s ${"falseAl"}%8s ${"miss"}%7s ${"AUC-ROC"}%8s ${"AUC-PR"}%8s")
    results.foreach { case (_, b, _) =>
      out(f"${b.name}%-28s ${b.precision}%7.4f ${b.recall}%7.4f ${b.f1}%7.4f ${b.falseAlarm}%8.4f ${b.miss}%7.4f " +
          f"${b.auc.map(a => f"$a%8.4f").getOrElse("     n/a")}%8s ${b.aucPr.map(a => f"$a%8.4f").getOrElse("     n/a")}%8s")
    }
    out("falseAl = normal flows flagged as attacks; miss = attack flows classified as normal")

    writeFile("results/ml_model_comparison.csv",
      Seq("model,train_accuracy,train_weighted_f1,test_accuracy,test_weighted_precision,test_weighted_recall,test_weighted_f1,test_macro_f1,binary_attack_precision,binary_attack_recall,binary_attack_f1,binary_false_alarm_rate,binary_miss_rate,binary_auc_roc,fit_seconds") ++
      results.map { case (m, b, _) =>
        Seq(m.name, m.trainAcc, m.trainWF1, m.acc, m.wPrec, m.wRec, m.wF1, m.macroF,
            b.precision, b.recall, b.f1, b.falseAlarm, b.miss, b.auc.getOrElse(Double.NaN), m.fitSec).mkString(",")
      })

    // ---- Per-class results of the primary model ---------------------------
    header("6. Per-class results - " + primaryName + " (test split)")
    val mP = mcMetrics(primary._3)
    out(f"${"class"}%-28s ${"test n"}%7s ${"prec"}%7s ${"recall"}%7s ${"F1"}%7s")
    val perClassLines = scala.collection.mutable.ArrayBuffer("class,test_flows,precision,recall,f1")
    allLabels.sortBy(l => -testCounts.getOrElse(l, 0L)).foreach { l =>
      val n = testCounts.getOrElse(l, 0L)
      out(f"${labelNames(l)}%-28s $n%7d ${mP.precision(l)}%7.4f ${mP.recall(l)}%7.4f ${mP.fMeasure(l)}%7.4f")
      perClassLines += s"${labelNames(l)},$n,${mP.precision(l)},${mP.recall(l)},${mP.fMeasure(l)}"
    }
    writeFile("results/ml_per_class_metrics.csv", perClassLines)

    // ---- Confusion matrix --------------------------------------------------
    header("7. Confusion matrix - " + primaryName + " (rows = actual, columns = predicted)")
    val cm = mP.confusionMatrix
    val cmLabels = mP.labels
    out("Legend: " + cmLabels.map(l => s"${l.toInt}=${labelNames(l)}").mkString(", "))
    out(f"${""}%5s" + cmLabels.map(l => f"${l.toInt}%7d").mkString)
    cmLabels.indices.foreach { i =>
      out(f"${cmLabels(i).toInt}%5d" + cmLabels.indices.map(j => f"${cm(i, j).toInt}%7d").mkString)
    }
    writeFile("results/ml_confusion_matrix.csv",
      Seq("actual\\predicted," + cmLabels.map(l => labelNames(l)).mkString(",")) ++
      cmLabels.indices.map(i => labelNames(cmLabels(i)) + "," + cmLabels.indices.map(j => cm(i, j).toInt).mkString(",")))

    val confusions = for {
      i <- cmLabels.indices; j <- cmLabels.indices if i != j && cm(i, j) > 0
    } yield (labelNames(cmLabels(i)), labelNames(cmLabels(j)), cm(i, j).toInt)
    out("Largest errors (actual -> predicted : flows):")
    if (confusions.isEmpty) out("  none: the model made no errors on the test split")
    confusions.sortBy(-_._3).take(10).foreach { case (a, p, n) => out(f"  $a%-28s -> $p%-28s : $n%d") }

    // ======================================================================
    // 8. Feature importance and coefficients
    // ======================================================================
    header("8. Feature importance - Random Forest (w)")
    val rfImp = rfModel.featureImportances.toArray
    val rfRanked = featureNames.zip(rfImp).sortBy(-_._2)
    out("Top 15 individual features:")
    rfRanked.take(15).foreach { case (n, v) => out(f"  $n%-40s ${v}%.4f") }

    val grouped = featureNames.zip(rfImp).groupBy(x => groupOf(x._1))
      .map { case (g, xs) => (g, xs.map(_._2).sum) }.toSeq.sortBy(-_._2)
    out("Top 10 after summing the one-hot slots of each categorical column:")
    grouped.take(10).foreach { case (n, v) => out(f"  $n%-40s ${v}%.4f") }
    writeFile("results/ml_feature_importance.csv",
      Seq("feature,rf_importance") ++ rfRanked.map { case (n, v) => s"$n,$v" })

    header("9. Decision Tree (w) - features used by the tree")
    featureNames.zip(dtModel.featureImportances.toArray).sortBy(-_._2).take(10)
      .foreach { case (n, v) => out(f"  $n%-40s ${v}%.4f") }
    out("Tree depth: " + dtModel.depth + ", nodes: " + dtModel.numNodes)

    header("10. Logistic Regression (w) - mean |coefficient| over the 12 classes (inputs are standardized)")
    val coef = lrModel.coefficientMatrix
    val meanAbs = (0 until coef.numCols).map { j =>
      (0 until coef.numRows).map(i => math.abs(coef(i, j))).sum / coef.numRows
    }
    featureNames.zip(meanAbs).sortBy(-_._2).take(10).foreach { case (n, v) => out(f"  $n%-40s ${v}%.4f") }

    // ======================================================================
    // 11. How much of the test score can be explained by near-copies?
    // Phase 4 (Q7) showed that the large classes have very few distinct flow
    // shapes, so many test flows have an identical twin in the training split.
    // Here the test flows are split into "shape seen in training" and
    // "shape never seen in training" and scored separately.
    // ======================================================================
    header("11. Test accuracy on seen vs. unseen flow shapes - " + primaryName)

    val shapeNumeric = Seq("total_pkts", "flow_SYN_flag_count", "flow_RST_flag_count", "flow_ACK_flag_count")
      .flatMap(originalToSafe.get)
    val shapeCols = (Seq("proto", "service", "resp_port_type").filter(catCols.contains) ++ shapeNumeric).toArray
    out("Shape = " + shapeCols.map(c => safeToOriginal.getOrElse(c, c)).mkString(", "))

    val trainShapes = trainDf.select(shapeCols.map(c => col(c)): _*).distinct().withColumn("_seen", lit(1))
    val flagged = rfTestPreds.join(trainShapes, shapeCols.toSeq, "left")
      .withColumn("seen", col("_seen").isNotNull)
      .withColumn("correct", when(col("prediction") === col("label"), 1.0).otherwise(0.0))
      .cache()

    flagged.groupBy("seen").agg(count(lit(1)).alias("n"), avg("correct").alias("accuracy")).orderBy("seen")
      .collect().foreach { r =>
        out(f"  shape seen in training: ${r.getBoolean(0)}%-5s flows = ${r.getLong(1)}%6d   accuracy = ${r.getDouble(2)}%.4f")
      }
    out("Share of test flows with a training twin, and accuracy on the unseen ones, per class:")
    out(f"${"class"}%-28s ${"test n"}%7s ${"seen %"}%7s ${"acc unseen"}%11s ${"n unseen"}%9s")
    val perClassSeen = flagged.groupBy("Attack_type").agg(
      count(lit(1)).alias("n"),
      avg(col("seen").cast("double")).alias("seen_share"),
      sum(when(!col("seen"), 1).otherwise(0)).alias("n_unseen"),
      avg(when(!col("seen"), col("correct"))).alias("acc_unseen")
    ).orderBy(desc("n")).collect()
    val seenLines = scala.collection.mutable.ArrayBuffer("class,test_flows,seen_share,unseen_flows,accuracy_on_unseen")
    perClassSeen.foreach { r =>
      val accU = if (r.isNullAt(4)) Double.NaN else r.getDouble(4)
      out(f"${r.getString(0)}%-28s ${r.getLong(1)}%7d ${100.0 * r.getDouble(2)}%7.1f ${accU}%11.4f ${r.getLong(3)}%9d")
      seenLines += s"${r.getString(0)},${r.getLong(1)},${r.getDouble(2)},${r.getLong(3)},$accU"
    }
    writeFile("results/ml_seen_vs_unseen_shapes.csv", seenLines)

    // ======================================================================
    // 12. Save models and the text log
    // ======================================================================
    header("12. Saving")
    prepModel.write.overwrite().save("models/prep_pipeline")
    rfModel.write.overwrite().save("models/random_forest")
    out("Saved: models/prep_pipeline, models/random_forest")
    out("Saved: results/ml_model_comparison.csv, ml_per_class_metrics.csv, ml_confusion_matrix.csv,")
    out("        ml_feature_importance.csv, ml_seen_vs_unseen_shapes.csv, ml_metrics.txt")
    writeFile("results/ml_metrics.txt", Seq(log.toString))

    spark.stop()
  }
}
