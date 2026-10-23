import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.hadoop.fs.{FileSystem, Path}

// ==========================================================================
// Phase 4 - Spark SQL Operations (RT-IoT2022)
// Input : data/analysis_dataset.parquet  (real units, from Phase 2)
//         data/attack_family_csv         (lookup table, from Phase 2)
// Output: results/sql_q1 ... results/sql_q7 (one CSV per query)
//         + console output for the report
//
// Views : flows          - one row per network flow (measured data only)
//         attack_family  - 12-row lookup: Attack_type -> family, class
//
// SQL features used: GROUP BY, HAVING, WHERE, ORDER BY, LIMIT, JOIN
//   (inner, USING, CROSS JOIN), CTEs (WITH), subqueries, CASE WHEN,
//   window functions (RANK, ROW_NUMBER, SUM OVER, running total),
//   statistical functions (percentile, STDDEV, COUNT DISTINCT)
// ==========================================================================

object SQLOperations {

  def header(title: String): Unit = {
    println("=" * 70)
    println(title)
    println("=" * 70)
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("RT-IoT2022 SQL Operations")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")
    import spark.implicits._

    val fs = FileSystem.get(sc.hadoopConfiguration)

    // Runs one query: prints the question, the SQL text and the result,
    // then saves the result as a single CSV file for the report
    def runQuery(id: String, question: String, sqlText: String, rows: Int = 20): DataFrame = {
      header(s"$id. $question")
      println(sqlText.trim)
      println("-" * 70)
      val result = spark.sql(sqlText).cache()
      result.show(rows, false)

      val path = s"results/sql_$id"
      val p = new Path(path)
      if (fs.exists(p)) fs.delete(p, true)
      result.coalesce(1).write.option("header", "true").csv(path)
      result
    }

    // ======================================================================
    // 0. Load the data and register the temporary views
    // ======================================================================
    header("0. Load data and register temporary views")

    val analysis = spark.read.parquet("data/analysis_dataset.parquet")

    // The flows view keeps only the measured flow data. The family and
    // traffic class live in their own lookup view, so queries that need
    // them must JOIN on Attack_type (star-schema style: fact + dimension).
    val flowsDf = analysis.drop("attack_family", "traffic_class", "is_attack")
    flowsDf.cache()
    flowsDf.createOrReplaceTempView("flows")

    // Lookup table saved by Phase 2. If it is missing, rebuild the same
    // 12 rows so the script still runs on its own.
    val familyDf =
      if (fs.exists(new Path("data/attack_family_csv")))
        spark.read.option("header", "true").csv("data/attack_family_csv")
      else Seq(
        ("MQTT_Publish",               "IoT device traffic",           "Normal"),
        ("Thing_Speak",                "IoT device traffic",           "Normal"),
        ("Wipro_bulb",                 "IoT device traffic",           "Normal"),
        ("DOS_SYN_Hping",              "Denial of service (DoS/DDoS)", "Attack"),
        ("DDOS_Slowloris",             "Denial of service (DoS/DDoS)", "Attack"),
        ("NMAP_UDP_SCAN",              "Reconnaissance",               "Attack"),
        ("NMAP_XMAS_TREE_SCAN",        "Reconnaissance",               "Attack"),
        ("NMAP_OS_DETECTION",          "Reconnaissance",               "Attack"),
        ("NMAP_TCP_scan",              "Reconnaissance",               "Attack"),
        ("NMAP_FIN_SCAN",              "Reconnaissance",               "Attack"),
        ("Metasploit_Brute_Force_SSH", "Brute force",                  "Attack"),
        ("ARP_poisioning",             "Spoofing (man-in-the-middle)", "Attack")
      ).toDF("Attack_type", "attack_family", "traffic_class")
    familyDf.createOrReplaceTempView("attack_family")

    println("flows view:         " + flowsDf.count() + " rows, " + flowsDf.columns.length + " columns")
    println("attack_family view: " + familyDf.count() + " rows")

    // Sanity check: the inner join used below must not drop any flow
    val joinedCount = spark.sql(
      "SELECT COUNT(*) FROM flows f JOIN attack_family a ON f.Attack_type = a.Attack_type"
    ).first().getLong(0)
    println("Rows after joining flows with attack_family: " + joinedCount +
            (if (joinedCount == flowsDf.count()) "  -> OK, no flows lost" else "  -> MISMATCH"))

    // ======================================================================
    // Q1 - How imbalanced are the classes, and what does a majority-class
    //      baseline score? (CTE + JOIN + window functions)
    // ======================================================================
    runQuery("q1_class_imbalance",
      "Class imbalance and majority-class baseline",
      """
      WITH class_counts AS (
        SELECT Attack_type, COUNT(*) AS flows
        FROM flows
        GROUP BY Attack_type
      )
      SELECT RANK() OVER (ORDER BY c.flows DESC)                     AS size_rank,
             c.Attack_type,
             a.attack_family,
             c.flows,
             ROUND(100.0 * c.flows / SUM(c.flows) OVER (), 2)        AS pct_of_flows,
             ROUND(100.0 * SUM(c.flows) OVER (ORDER BY c.flows DESC
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
                   / SUM(c.flows) OVER (), 2)                        AS cumulative_pct,
             ROUND(MAX(c.flows) OVER () / c.flows, 1)                AS majority_to_class_ratio,
             CAST(ROUND(c.flows * 0.3) AS INT)                       AS approx_test_rows_30pct
      FROM class_counts c
      JOIN attack_family a ON c.Attack_type = a.Attack_type
      ORDER BY c.flows DESC
      """)

    // ======================================================================
    // Q2 - How do the attack families differ in flow size and duration
    //      once skew is taken into account? (JOIN + GROUP BY + percentiles)
    // ======================================================================
    runQuery("q2_family_statistics",
      "Flow size and duration per attack family (mean vs median vs p95)",
      """
      SELECT a.attack_family,
             a.traffic_class,
             COUNT(*)                                       AS flows,
             COUNT(DISTINCT f.Attack_type)                  AS classes,
             ROUND(AVG(f.flow_duration), 3)                 AS avg_duration_s,
             ROUND(percentile(f.flow_duration, 0.5), 4)     AS median_duration_s,
             ROUND(percentile(f.flow_duration, 0.95), 3)    AS p95_duration_s,
             ROUND(AVG(f.total_pkts), 2)                    AS avg_pkts,
             percentile(f.total_pkts, 0.5)                  AS median_pkts,
             ROUND(STDDEV(f.total_pkts), 2)                 AS stddev_pkts,
             ROUND(AVG(f.payload_bytes_per_second), 1)      AS avg_payload_bytes_per_s
      FROM flows f
      JOIN attack_family a USING (Attack_type)
      GROUP BY a.attack_family, a.traffic_class
      ORDER BY flows DESC
      """)

    // ======================================================================
    // Q3 - Which services dominate each family? (CTE + JOIN + ROW_NUMBER
    //      ranking inside each family + share of the family total)
    // ======================================================================
    runQuery("q3_top_services_per_family",
      "Top 3 services in each attack family (window ranking)",
      """
      WITH service_counts AS (
        SELECT a.attack_family, f.service, COUNT(*) AS flows
        FROM flows f
        JOIN attack_family a USING (Attack_type)
        GROUP BY a.attack_family, f.service
      ),
      ranked AS (
        SELECT attack_family, service, flows,
               ROW_NUMBER() OVER (PARTITION BY attack_family ORDER BY flows DESC, service) AS rank_in_family,
               ROUND(100.0 * flows / SUM(flows) OVER (PARTITION BY attack_family), 2) AS pct_of_family
        FROM service_counts
      )
      SELECT attack_family, rank_in_family, service, flows, pct_of_family
      FROM ranked
      WHERE rank_in_family <= 3
      ORDER BY attack_family, rank_in_family
      """, rows = 30)

    // ======================================================================
    // Q4 - Which protocol / destination-port combinations are almost purely
    //      attack traffic, and which class dominates each one?
    //      (CTE + JOIN + CASE + window + HAVING)
    // ======================================================================
    runQuery("q4_proto_port_purity",
      "Attack share and dominant class per protocol / destination port type",
      """
      WITH combo_class AS (
        SELECT f.proto, f.resp_port_type, f.Attack_type, a.traffic_class,
               COUNT(*) AS flows
        FROM flows f
        JOIN attack_family a USING (Attack_type)
        GROUP BY f.proto, f.resp_port_type, f.Attack_type, a.traffic_class
      ),
      with_rank AS (
        SELECT *,
               ROW_NUMBER() OVER (PARTITION BY proto, resp_port_type ORDER BY flows DESC, Attack_type) AS rn,
               SUM(flows)   OVER (PARTITION BY proto, resp_port_type)                    AS combo_flows
        FROM combo_class
      )
      SELECT proto,
             resp_port_type,
             SUM(flows)                                                               AS flows,
             COUNT(*)                                                                 AS classes_present,
             ROUND(100.0 * SUM(CASE WHEN traffic_class = 'Attack' THEN flows ELSE 0 END)
                   / SUM(flows), 2)                                                   AS pct_attack,
             MAX(CASE WHEN rn = 1 THEN Attack_type END)                               AS dominant_class,
             ROUND(100.0 * MAX(CASE WHEN rn = 1 THEN flows END) / MAX(combo_flows), 2) AS dominant_pct
      FROM with_rank
      GROUP BY proto, resp_port_type
      HAVING SUM(flows) >= 100
      ORDER BY flows DESC
      """)

    // ======================================================================
    // Q5 - Phase 3 found two look-alike pairs (same packets, duration and
    //      SYN/RST ratio). Which other features tell them apart?
    //      (WHERE IN + CASE + GROUP BY + COUNT DISTINCT)
    // ======================================================================
    runQuery("q5_lookalike_pairs",
      "Features that separate the look-alike class pairs from Phase 3",
      """
      SELECT CASE WHEN Attack_type IN ('DOS_SYN_Hping', 'NMAP_TCP_scan')
                  THEN 'A: SYN flood vs TCP scan'
                  ELSE 'B: XMAS scan vs OS detection' END     AS pair,
             Attack_type,
             COUNT(*)                                          AS flows,
             ROUND(AVG(flow_SYN_flag_count), 2)                AS avg_syn,
             ROUND(AVG(flow_RST_flag_count), 2)                AS avg_rst,
             ROUND(AVG(flow_ACK_flag_count), 2)                AS avg_ack,
             ROUND(AVG(flow_FIN_flag_count), 2)                AS avg_fin,
             ROUND(AVG(fwd_init_window_size), 1)               AS avg_fwd_init_win,
             COUNT(DISTINCT fwd_init_window_size)              AS distinct_fwd_init_win,
             ROUND(AVG(fwd_header_size_max), 1)                AS avg_fwd_header_max,
             ROUND(100.0 * AVG(CASE WHEN orig_port_type = 'well_known' THEN 1 ELSE 0 END), 1) AS pct_orig_well_known,
             ROUND(100.0 * AVG(CASE WHEN resp_port_type = 'well_known' THEN 1 ELSE 0 END), 1) AS pct_resp_well_known
      FROM flows
      WHERE Attack_type IN ('DOS_SYN_Hping', 'NMAP_TCP_scan',
                            'NMAP_XMAS_TREE_SCAN', 'NMAP_OS_DETECTION')
      GROUP BY Attack_type
      ORDER BY pair, flows DESC
      """)

    // ======================================================================
    // Q6 - Are the heaviest flows (top 1% by packets) attacks?
    //      (CTE + scalar percentile + CROSS JOIN + subquery + window)
    // ======================================================================
    val p99 = spark.sql("SELECT percentile(total_pkts, 0.99) FROM flows").first().getDouble(0)
    println(f"99th percentile of total_pkts = $p99%.1f packets")

    runQuery("q6_heaviest_flows",
      "Which classes produce the top 1% heaviest flows?",
      """
      WITH threshold AS (
        SELECT percentile(total_pkts, 0.99) AS p99 FROM flows
      ),
      class_sizes AS (
        SELECT Attack_type, COUNT(*) AS class_flows FROM flows GROUP BY Attack_type
      )
      SELECT f.Attack_type,
             a.traffic_class,
             COUNT(*)                                             AS heavy_flows,
             ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2)   AS pct_of_heavy_flows,
             c.class_flows,
             ROUND(100.0 * COUNT(*) / c.class_flows, 2)           AS pct_of_class,
             MAX(f.total_pkts)                                    AS max_pkts
      FROM flows f
      CROSS JOIN threshold t
      JOIN attack_family a USING (Attack_type)
      JOIN class_sizes  c USING (Attack_type)
      WHERE f.total_pkts > t.p99
      GROUP BY f.Attack_type, a.traffic_class, c.class_flows
      ORDER BY heavy_flows DESC
      """)

    // ======================================================================
    // Q7 - How uniform is each class? Few distinct flow "shapes" mean the
    //      class is easy to learn but also repetitive (risk of memorising).
    //      (GROUP BY + multi-column COUNT DISTINCT + STDDEV / AVG)
    // ======================================================================
    runQuery("q7_class_homogeneity",
      "Distinct flow shapes and variability per class",
      """
      SELECT Attack_type,
             COUNT(*)                                                      AS flows,
             COUNT(DISTINCT proto, service, resp_port_type, total_pkts,
                   flow_SYN_flag_count, flow_RST_flag_count,
                   flow_ACK_flag_count)                                    AS distinct_shapes,
             ROUND(100.0 * COUNT(DISTINCT proto, service, resp_port_type, total_pkts,
                   flow_SYN_flag_count, flow_RST_flag_count,
                   flow_ACK_flag_count) / COUNT(*), 2)                     AS pct_distinct,
             ROUND(STDDEV(total_pkts)    / NULLIF(AVG(total_pkts), 0), 2)    AS cv_pkts,
             ROUND(STDDEV(flow_duration) / NULLIF(AVG(flow_duration), 0), 2) AS cv_duration
      FROM flows
      GROUP BY Attack_type
      ORDER BY pct_distinct ASC
      """)

    header("Saved results")
    println("results/sql_q1_class_imbalance ... results/sql_q7_class_homogeneity (one CSV each)")

    spark.stop()
  }
}
