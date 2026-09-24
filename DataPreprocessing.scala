import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.Column
import org.apache.spark.sql.expressions.Window
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{NumericType, StringType}
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, StringIndexerModel, OneHotEncoder}
import org.apache.spark.ml.stat.Correlation
import org.apache.spark.ml.linalg.Matrix

object DataPreprocessing {

  def main(args: Array[String]): Unit = {

    // Create Spark session
    val spark = SparkSession.builder()
      .appName("RT-IoT2022 Data Preprocessing")
      .master("local[*]")
      .getOrCreate()

    // Read the dataset
    val df = spark.read
      .option("header", "true")
      .option("inferSchema", "true")
      .csv("RT_IOT2022.csv")
      .drop("_c0")

    // Show basic dataset information
    println("======================================")
    println("RT-IoT2022 Dataset")
    println("======================================")

    println("Number of rows: " + df.count())
    println("Number of columns: " + df.columns.length)

    // Display column names
    println("Column names:")
    df.columns.foreach(println)

    // Display schema
    println("======================================")
    println("Dataset Schema")
    println("======================================")

    df.printSchema()

    // Display first 5 rows
    println("======================================")
    println("First 5 Rows")
    println("======================================")

    df.show(5, false)
    // Check for missing values
    println("======================================")
    println("Missing Values")
    println("======================================")

    val missingValues = df.columns.map { c =>
      sum(col(s"`$c`").isNull.cast("int")).alias(c)
    }

    df.select(missingValues: _*).show(false)
    // Check for duplicate rows
    println("======================================")
    println("Duplicate Rows")
    println("======================================")

    val totalRows = df.count()
    val distinctRows = df.distinct().count()
    val duplicateRows = totalRows - distinctRows

    println("Total rows: " + totalRows)
    println("Distinct rows: " + distinctRows)
    println("Duplicate rows: " + duplicateRows)
    val cleanedDf = df.dropDuplicates().cache()

    println("======================================")
    println("After Removing Duplicates")
    println("======================================")

    println("Rows after cleaning: " + cleanedDf.count())
    println("Columns after cleaning: " + cleanedDf.columns.length)

    // Check for negative values
    println("======================================")
    println("Negative Values")
    println("======================================")

    val numericColumns = cleanedDf.columns.filter { c =>
      cleanedDf.schema(c).dataType.isInstanceOf[NumericType]
    }

    numericColumns.foreach { c =>
      val count = cleanedDf
        .filter(col(s"`$c`") < 0)
        .count()

      if (count > 0) {
        println(c + ": " + count)
      }
    }

    // Check for outliers using IQR
    println("======================================")
    println("Outliers (IQR Method)")
    println("======================================")

    numericColumns.foreach { c =>

      val tempDf = cleanedDf
        .select(col(s"`$c`").alias("value"))

      val quantiles = tempDf.stat.approxQuantile(
        "value",
        Array(0.25, 0.75),
        0.01
      )

      val q1 = quantiles(0)
      val q3 = quantiles(1)
      val iqr = q3 - q1
      val lowerBound = q1 - 1.5 * iqr
      val upperBound = q3 + 1.5 * iqr

      val outlierCount = tempDf
        .filter(
          (col("value") < lowerBound) ||
          (col("value") > upperBound)
        )
        .count()

      if (outlierCount > 0) {
        println(c + ": " + outlierCount + " outliers")
      }
    }
    // ======================================
    // Data Reduction - Step 1: Constant Features
    // ======================================

    println("======================================")
    println("Data Reduction - Constant Features")
    println("======================================")

    // Exclude the target column from feature reduction
    val featureColumns = cleanedDf.columns.filter(_ != "Attack_type")

    // Identify features that contain only one unique value
    val constantColumns = featureColumns.filter { c =>
      cleanedDf
        .select(col(s"`$c`"))
        .distinct()
        .limit(2)
        .count() <= 1
    }

    println("Number of features before reduction: " + featureColumns.length)
    println("Number of constant features found: " + constantColumns.length)

    if (constantColumns.nonEmpty) {
      println("Constant features:")
      constantColumns.foreach(println)
    } else {
      println("No constant features found.")
    }
    // Remove constant features
    val reducedDf = cleanedDf.drop(constantColumns: _*)

    println("======================================")
    println("After Constant Feature Removal")
    println("======================================")

    println("Rows after reduction: " + reducedDf.count())
    println("Columns after reduction: " + reducedDf.columns.length)

    println("Removed features:")
    constantColumns.foreach(println)

    // ======================================
    // Data Reduction - Step 2: Correlation-Based Feature Selection
    // ======================================

    println("======================================")
    println("Data Reduction - Correlation Analysis")
    println("======================================")

    // Select numeric features only
    val correlationColumns = reducedDf.columns.filter { c =>
      reducedDf.schema(c).dataType.isInstanceOf[NumericType]
    }

    println(
      "Numeric features used for correlation: " +
      correlationColumns.length
    )

    // Rename columns temporarily because VectorAssembler
    // interprets dots in column names as nested fields
    val safeCorrelationColumns =
      correlationColumns.indices.map(i => s"corr_$i").toArray

    val safeNumericDf = reducedDf.select(
      correlationColumns
        .zip(safeCorrelationColumns)
        .map { case (original, safe) =>
          col(s"`$original`").cast("double").alias(safe)
        }: _*
    )

    // Assemble numeric features into one vector
    val correlationAssembler = new VectorAssembler()
      .setInputCols(safeCorrelationColumns)
      .setOutputCol("features")

    val correlationVectorDf =
      correlationAssembler
        .transform(safeNumericDf)
        .select("features")

    // Calculate Pearson correlation matrix
    val correlationMatrix =
      Correlation
        .corr(correlationVectorDf, "features", "pearson")
        .head()
        .getAs[Matrix](0)

    // Detect highly correlated feature pairs
    val correlationThreshold = 0.95

    val highlyCorrelatedPairs =
      for {
        i <- correlationColumns.indices
        j <- (i + 1) until correlationColumns.length
        corr = correlationMatrix(i, j)
        if !corr.isNaN &&
           math.abs(corr) >= correlationThreshold
      } yield (
        correlationColumns(i),
        correlationColumns(j),
        corr
      )

    println(
      "Number of highly correlated pairs: " +
      highlyCorrelatedPairs.length
    )

    highlyCorrelatedPairs
      .sortBy(x => -math.abs(x._3))
      .foreach { case (feature1, feature2, corr) =>
        println(
          feature1 + " <-> " +
          feature2 + " : " +
          f"$corr%.4f"
        )
      }

    // Features selected for removal after inspecting highly correlated groups.
    // A threshold of |r| >= 0.95 was used to identify candidate pairs.
    // Only the strongest redundant relationships (approximately |r| >= 0.99)
    // were selected for removal to avoid excessive information loss.
    val correlationDropColumns = Array(
      "flow_iat.tot",
      "fwd_iat.tot",
      "idle.tot",
      "fwd_pkts_per_sec",
      "bwd_pkts_per_sec",
      "bwd_bulk_packets",
      "fwd_iat.max",
      "idle.max",
      "idle.avg",
      "bwd_data_pkts_tot",
      "bwd_pkts_payload.tot"
    )

    // Remove redundant correlated features
    val finalReducedDf =
      reducedDf.drop(correlationDropColumns: _*)

    println("======================================")
    println("After Correlation-Based Reduction")
    println("======================================")

    println(
      "Rows after correlation reduction: " +
      finalReducedDf.count()
    )

    println(
      "Columns after correlation reduction: " +
      finalReducedDf.columns.length
    )

    println(
      "Correlation-based features removed: " +
      correlationDropColumns.length
    )

    println("Removed correlated features:")
    correlationDropColumns.foreach(println)

    println(
      "Final predictive features: " +
      finalReducedDf.columns.count(_ != "Attack_type")
    )

    // ==========================================================
    // DATA TRANSFORMATION
    // ----------------------------------------------------------
    // T1. Data type conversion and consistent categorical formats
    // T2. Feature engineering (domain-relevant features)
    // T3. Log transformation of highly skewed features
    // T4. Encoding categorical variables (StringIndexer + OneHot)
    // T5. Standardization (Z-score)
    // T6. Save the final datasets + 15-row snapshot
    // ==========================================================

    val columnsBeforeTransformation = finalReducedDf.columns.length

    // ----------------------------------------------------------
    // T1. Data types and consistent categorical formats
    // ----------------------------------------------------------
    println("======================================")
    println("Transformation T1 - Data Types and Formats")
    println("======================================")

    val stringColumns = finalReducedDf.schema.fields
      .filter(_.dataType == StringType)
      .map(_.name)

    println("Categorical (string) columns: " + stringColumns.mkString(", "))

    // Category distributions before cleaning up their format
    println("proto values:")
    finalReducedDf.groupBy("proto").count().orderBy(desc("count")).show(false)
    println("service values:")
    finalReducedDf.groupBy("service").count().orderBy(desc("count")).show(false)

    // proto/service: trim and lowercase so the same category is never
    // spelled two ways. In this dataset "-" in service means the service
    // could not be identified, so it becomes an explicit "unknown" category
    // instead of an unlabeled symbol.
    var transformedDf = finalReducedDf
      .withColumn("proto", lower(trim(col("proto"))))
      .withColumn(
        "service",
        when(col("service").isNull || trim(col("service")) === "-", "unknown")
          .otherwise(lower(trim(col("service"))))
      )

    val unknownServiceCount =
      transformedDf.filter(col("service") === "unknown").count()
    println("Rows with service '-' recoded as 'unknown': " + unknownServiceCount)

    // inferSchema reads counts as integer and rates as double.
    // All numeric features are cast to double so every feature has
    // one consistent type for scaling and for MLlib.
    val numericBeforeCast = transformedDf.schema.fields
      .filter(_.dataType.isInstanceOf[NumericType])
      .map(_.name)

    val nonDoubleCount = transformedDf.schema.fields
      .count(f => f.dataType.isInstanceOf[NumericType] &&
                  f.dataType.typeName != "double")
    println("Numeric columns cast from integer/long to double: " + nonDoubleCount)

    transformedDf = transformedDf.select(
      transformedDf.columns.map { c =>
        if (numericBeforeCast.contains(c)) col(s"`$c`").cast("double").alias(c)
        else col(s"`$c`")
      }: _*
    )

    // ----------------------------------------------------------
    // T2. Feature engineering
    // ----------------------------------------------------------
    println("======================================")
    println("Transformation T2 - Feature Engineering")
    println("======================================")

    // Port numbers are identifiers, not quantities (port 443 is not
    // "bigger" than port 80), so using them as raw numbers misleads the
    // model. They are grouped into the standard IANA ranges instead.
    def portCategory(port: Column): Column =
      when(port < 1024, "well_known")
        .when(port < 49152, "registered")
        .otherwise("dynamic")

    transformedDf = transformedDf
      .withColumn("orig_port_type", portCategory(col("`id.orig_p`")))
      .withColumn("resp_port_type", portCategory(col("`id.resp_p`")))
      .drop("id.orig_p", "id.resp_p")

    // Packet-level features that describe the shape of a flow:
    //  - total_pkts:     overall size of the flow
    //  - fwd_pkt_ratio:  share of packets sent by the originator
    //                    (scans and floods are almost entirely one-directional)
    //  - syn_flag_ratio: SYN flags per packet (SYN floods and TCP scans)
    //  - rst_flag_ratio: RST flags per packet (closed ports answering scans)
    val totalPkts = col("fwd_pkts_tot") + col("bwd_pkts_tot")

    def safeRatio(numerator: Column): Column =
      when(totalPkts > 0, numerator / totalPkts).otherwise(0.0)

    transformedDf = transformedDf
      .withColumn("total_pkts", totalPkts)
      .withColumn("fwd_pkt_ratio", safeRatio(col("fwd_pkts_tot")))

    if (transformedDf.columns.contains("flow_SYN_flag_count"))
      transformedDf = transformedDf
        .withColumn("syn_flag_ratio", safeRatio(col("flow_SYN_flag_count")))

    if (transformedDf.columns.contains("flow_RST_flag_count"))
      transformedDf = transformedDf
        .withColumn("rst_flag_ratio", safeRatio(col("flow_RST_flag_count")))

    // Binary target: normal IoT traffic vs. attack.
    // The three normal-traffic classes come from the dataset documentation;
    // the class list printed below should be checked against these names.
    val normalTrafficClasses = Seq("MQTT_Publish", "Thing_Speak", "Wipro_bulb")

    println("Attack_type classes:")
    transformedDf.groupBy("Attack_type").count().orderBy(desc("count")).show(false)

    transformedDf = transformedDf.withColumn(
      "is_attack",
      when(col("Attack_type").isin(normalTrafficClasses: _*), 0.0).otherwise(1.0)
    )

    println("Normal vs attack rows:")
    transformedDf.groupBy("is_attack").count().orderBy("is_attack").show(false)

    val engineeredFeatures = Seq(
      "orig_port_type", "resp_port_type", "total_pkts", "fwd_pkt_ratio",
      "syn_flag_ratio", "rst_flag_ratio", "is_attack"
    ).filter(transformedDf.columns.contains)

    println("Engineered features: " + engineeredFeatures.mkString(", "))
    println("Raw port columns replaced by port categories: id.orig_p, id.resp_p")

    transformedDf = transformedDf.cache()

    // Copy with REAL units (before log transform, encoding and scaling)
    // for the RDD and SQL phases, so results stay easy to interpret
    // (e.g. average flow_duration in seconds, not log-seconds).
    val analysisDf = transformedDf

    // ----------------------------------------------------------
    // T3. Log transformation of highly skewed features
    // ----------------------------------------------------------
    println("======================================")
    println("Transformation T3 - Log Transformation")
    println("======================================")

    // The IQR check found many outliers. In network traffic these are
    // real behaviour (floods create huge packet counts), so they are kept
    // rather than removed. log1p compresses the long tail so a few extreme
    // flows do not dominate distance- or gradient-based models.
    // Applied only to non-negative features with skewness > 1.
    val continuousColumns = transformedDf.schema.fields
      .filter(_.dataType.isInstanceOf[NumericType])
      .map(_.name)
      .filter(_ != "is_attack")

    val shapeStats = transformedDf.select(
      continuousColumns.zipWithIndex.flatMap { case (c, i) =>
        Seq(
          min(col(s"`$c`")).alias(s"min_$i"),
          skewness(col(s"`$c`")).alias(s"skew_$i")
        )
      }: _*
    ).first()

    val logColumns = continuousColumns.zipWithIndex.collect {
      case (c, i)
        if !shapeStats.isNullAt(2 * i) &&
           !shapeStats.isNullAt(2 * i + 1) &&
           shapeStats.getDouble(2 * i) >= 0.0 &&
           shapeStats.getDouble(2 * i + 1) > 1.0 => c
    }

    println("Features log-transformed: " + logColumns.length +
            " of " + continuousColumns.length + " numeric features")
    logColumns.foreach { c =>
      val i = continuousColumns.indexOf(c)
      println(f"  $c%-35s skewness before = ${shapeStats.getDouble(2 * i + 1)}%.2f")
    }

    transformedDf = transformedDf.select(
      transformedDf.columns.map { c =>
        if (logColumns.contains(c)) log1p(col(s"`$c`")).alias(c)
        else col(s"`$c`")
      }: _*
    )

    val skewAfter = transformedDf.select(
      logColumns.zipWithIndex.map { case (c, i) =>
        skewness(col(s"`$c`")).alias(s"skew_$i")
      }: _*
    ).first()

    println("Skewness after log transformation:")
    logColumns.zipWithIndex.foreach { case (c, i) =>
      if (!skewAfter.isNullAt(i))
        println(f"  $c%-35s skewness after  = ${skewAfter.getDouble(i)}%.2f")
    }

    // ----------------------------------------------------------
    // T4. Encoding categorical variables
    // ----------------------------------------------------------
    println("======================================")
    println("Transformation T4 - Categorical Encoding")
    println("======================================")

    // MLlib needs numbers, not strings.
    // - Input categories: StringIndexer -> OneHotEncoder, because these
    //   categories have no natural order (tcp is not "less than" udp).
    // - Target (Attack_type): StringIndexer only -> "label" column
    //   (index 0 = most frequent class).
    // handleInvalid = "keep" gives unseen categories their own index
    // instead of crashing when the model is applied to new data.
    val categoricalColumns =
      Array("proto", "service", "orig_port_type", "resp_port_type")

    val featureIndexers: Array[PipelineStage] = categoricalColumns.map { c =>
      new StringIndexer()
        .setInputCol(c)
        .setOutputCol(c + "_idx")
        .setHandleInvalid("keep")
    }

    val labelIndexer = new StringIndexer()
      .setInputCol("Attack_type")
      .setOutputCol("label")
      .setStringOrderType("frequencyDesc")

    val oneHotEncoder = new OneHotEncoder()
      .setInputCols(categoricalColumns.map(_ + "_idx"))
      .setOutputCols(categoricalColumns.map(_ + "_vec"))
      .setHandleInvalid("keep")

    val encodingModel = new Pipeline()
      .setStages(featureIndexers ++ Array[PipelineStage](labelIndexer, oneHotEncoder))
      .fit(transformedDf)

    transformedDf = encodingModel.transform(transformedDf)

    // Print the category -> index mappings for the report
    categoricalColumns.indices.foreach { i =>
      val model = encodingModel.stages(i).asInstanceOf[StringIndexerModel]
      println(categoricalColumns(i) + " categories (" +
              model.labelsArray(0).length + "): " +
              model.labelsArray(0).zipWithIndex
                .map { case (l, idx) => s"$idx=$l" }.mkString(", "))
    }

    val labelModel = encodingModel.stages(categoricalColumns.length)
      .asInstanceOf[StringIndexerModel]
    println("Attack_type -> label mapping:")
    labelModel.labelsArray(0).zipWithIndex.foreach { case (l, idx) =>
      println(s"  $idx = $l")
    }

    // Keep the preprocessed dataset free of Spark ML internals:
    // the *_idx helper columns are only needed to build the one-hot vectors.
    transformedDf = transformedDf.drop(categoricalColumns.map(_ + "_idx"): _*)

    // Log-transformed + encoded, NOT scaled (used for ML; the ML phase
    // scales it after the train/test split).
    val preprocessedDf = transformedDf.cache()

    // ----------------------------------------------------------
    // T5. Standardization (Z-score)
    // ----------------------------------------------------------
    println("======================================")
    println("Transformation T5 - Standardization (Z-score)")
    println("======================================")

    // Features have very different ranges (flow_duration vs. flag counts),
    // so each continuous feature is rescaled to mean 0 and std 1.
    // All means and standard deviations are computed in ONE pass
    // instead of one Spark job per column.
    //
    // NOTE for Phase 5: to avoid data leakage, the ML phase should refit
    // StandardScaler on the TRAINING split only and read the unscaled
    // preprocessed dataset. The standardized dataset here shows the
    // transformation for Phase 2.
    val meanStdRow = preprocessedDf.select(
      continuousColumns.zipWithIndex.flatMap { case (c, i) =>
        Seq(
          avg(col(s"`$c`")).alias(s"mean_$i"),
          stddev(col(s"`$c`")).alias(s"std_$i")
        )
      }: _*
    ).first()

    // The scaling formulas are kept in a list so the exact same
    // transformation can be applied to the full data and to the snapshot.
    var scaledCount = 0
    val standardizeExprs: Array[Column] = preprocessedDf.columns.map { c =>
      val i = continuousColumns.indexOf(c)
      if (i >= 0 && !meanStdRow.isNullAt(2 * i + 1)) {
        val mean = meanStdRow.getDouble(2 * i)
        val std = meanStdRow.getDouble(2 * i + 1)
        if (std != 0.0 && !std.isNaN) {
          scaledCount += 1
          ((col(s"`$c`") - lit(mean)) / lit(std)).alias(c)
        } else col(s"`$c`")
      } else col(s"`$c`")
    }

    val standardizedDf = preprocessedDf.select(standardizeExprs: _*)

    println("Features standardized: " + scaledCount)

    // Before/after check on a few features
    val checkColumns = Seq("flow_duration", "total_pkts", "fwd_pkt_ratio")
      .filter(preprocessedDf.columns.contains)

    checkColumns.foreach { c =>
      val before = preprocessedDf
        .select(avg(col(s"`$c`")), stddev(col(s"`$c`"))).first()
      val after = standardizedDf
        .select(avg(col(s"`$c`")), stddev(col(s"`$c`"))).first()
      println(f"  $c%-20s before: mean=${before.getDouble(0)}%.4f std=${before.getDouble(1)}%.4f" +
              f"  |  after: mean=${after.getDouble(0)}%.4f std=${after.getDouble(1)}%.4f")
    }

    // ----------------------------------------------------------
    // T6. Summary, snapshot and output
    // ----------------------------------------------------------
    println("======================================")
    println("After Data Transformation")
    println("======================================")

    println("Rows after transformation: " + preprocessedDf.count())
    println("Columns before transformation: " + columnsBeforeTransformation)
    println("Columns after transformation: " + preprocessedDf.columns.length)

    println("Final schema:")
    preprocessedDf.printSchema()

    // Snapshot of the final dataset (Phase 2 deliverable).
    // One random row from EACH attack class (12 rows) so the snapshot
    // shows the variety of the data instead of 15 rows of one class.
    // Seed 42 keeps the same rows on every run.
    val classWindow = Window.partitionBy("Attack_type").orderBy(col("rand_key"))

    val snapshotRows = preprocessedDf
      .withColumn("rand_key", rand(42))
      .withColumn("row_in_class", row_number().over(classWindow))
      .filter(col("row_in_class") === 1)
      .drop("rand_key", "row_in_class")
      .orderBy("label")
      .cache()

    val snapshotColumns = Seq(
      "Attack_type", "label", "is_attack",
      "proto", "proto_vec", "service", "service_vec",
      "resp_port_type", "flow_duration", "total_pkts",
      "fwd_pkt_ratio", "syn_flag_ratio"
    ).filter(preprocessedDf.columns.contains)

    println("Snapshot of the preprocessed dataset (one row per class):")
    snapshotRows
      .select(snapshotColumns.map(c => col(s"`$c`")): _*)
      .show(20, false)

    // Same 12 rows after standardization
    println("Snapshot of the standardized dataset (same rows):")
    snapshotRows
      .select(standardizeExprs: _*)
      .select(snapshotColumns.map(c => col(s"`$c`")): _*)
      .show(20, false)

    // Save outputs
    analysisDf.write.mode("overwrite")
      .parquet("data/analysis_dataset.parquet")

    preprocessedDf.write.mode("overwrite")
      .parquet("data/preprocessed_dataset.parquet")

    standardizedDf.write.mode("overwrite")
      .parquet("data/standardized_dataset.parquet")

    // CSV cannot store vector columns, so the CSV snapshot drops them
    snapshotRows
      .drop(categoricalColumns.map(_ + "_vec"): _*)
      .coalesce(1)
      .write.mode("overwrite")
      .option("header", "true")
      .csv("results/preprocessed_snapshot_csv")

    println("Saved: data/analysis_dataset.parquet     (real units, for RDD and SQL)")
    println("Saved: data/preprocessed_dataset.parquet (log + encoded, unscaled, for ML)")
    println("Saved: data/standardized_dataset.parquet (log + encoded + Z-score scaled)")
    println("Saved: results/preprocessed_snapshot_csv (12-row snapshot, one per class)")

    spark.stop()
  }
}