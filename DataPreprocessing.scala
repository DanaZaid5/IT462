import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.ml.feature.VectorAssembler
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
      cleanedDf.schema(c).dataType.isInstanceOf[
        org.apache.spark.sql.types.NumericType
      ]
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
  reducedDf.schema(c).dataType.isInstanceOf[
    org.apache.spark.sql.types.NumericType
  ]
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

    // Standardization using Z-score
    println("======================================")
    println("Standardization")
    println("======================================")
  

var standardizedDf = finalReducedDf
val reducedNumericColumns = finalReducedDf.columns.filter { c =>
  finalReducedDf.schema(c).dataType.isInstanceOf[
    org.apache.spark.sql.types.NumericType
  ]
}

    reducedNumericColumns.foreach { c =>
      val stats = finalReducedDf.select(
        avg(col(s"`$c`")).alias("mean"),
        stddev(col(s"`$c`")).alias("stddev")
      ).first()

      val mean = stats.getAs[Double]("mean")
      val stddevValue = stats.getAs[Double]("stddev")

      if (stddevValue != 0.0 && !stddevValue.isNaN) {
        standardizedDf = standardizedDf.withColumn(
          c,
          (col(s"`$c`") - lit(mean)) / lit(stddevValue)
        )
      }
    }

    println("Standardization completed for numeric columns.")
    println("Rows after standardization: " + standardizedDf.count())
    println("Columns after standardization: " + standardizedDf.columns.length)

    println("First 5 standardized rows:")
    standardizedDf.show(5, false)
    spark.stop()
  }
}