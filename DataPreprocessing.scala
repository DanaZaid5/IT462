import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._

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
    val cleanedDf = df.dropDuplicates()

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

    // Standardization using Z-score
    println("======================================")
    println("Standardization")
    println("======================================")

    var standardizedDf = cleanedDf

    numericColumns.foreach { c =>
      val stats = cleanedDf.select(
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