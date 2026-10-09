import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._
import org.apache.spark.rdd.RDD
import org.apache.hadoop.fs.{FileSystem, Path}

// ==========================================================================
// Phase 3 - RDD Operations (RT-IoT2022)
// Input : data/analysis_dataset.parquet  (real units, from Phase 2)
//         data/attack_family_csv         (reference table, from Phase 2)
// Output: results/rdd_* (saveAsTextFile) + console output for the report
//
// Transformations used: map, filter, flatMap, reduceByKey, aggregateByKey,
//                       groupByKey, mapValues, join, distinct, sortBy,
//                       sortByKey, keys
// Actions used:         count, first, take, collect, reduce, countByKey,
//                       takeSample, histogram, foreach, saveAsTextFile
// ==========================================================================

// Case class lives outside main so Spark can serialize it
case class Flow(
  attackType: String,
  family: String,
  trafficClass: String,
  proto: String,
  service: String,
  origPortType: String,
  respPortType: String,
  duration: Double,   // seconds
  fwdPkts: Double,
  bwdPkts: Double,
  totalPkts: Double,
  fwdRatio: Double,   // share of packets sent by the originator
  synRatio: Double,   // SYN flags per packet
  rstRatio: Double    // RST flags per packet
)

object RDDOperations {

  def header(title: String): Unit = {
    println("=" * 60)
    println(title)
    println("=" * 60)
  }

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .appName("RT-IoT2022 RDD Operations")
      .master("local[*]")
      .getOrCreate()
    val sc = spark.sparkContext
    sc.setLogLevel("WARN")

    // Overwrite-safe saveAsTextFile (Spark fails if the folder exists)
    def saveText(rdd: RDD[String], path: String): Unit = {
      val fs = FileSystem.get(sc.hadoopConfiguration)
      val p = new Path(path)
      if (fs.exists(p)) fs.delete(p, true)
      rdd.coalesce(1).saveAsTextFile(path)
    }

    // ======================================================================
    // 0. Load: DataFrame -> RDD[Flow]
    // ======================================================================
    val raw = spark.read.parquet("data/analysis_dataset.parquet")

    def num(name: String) =
      if (raw.columns.contains(name))
        coalesce(col(s"`$name`").cast("double"), lit(0.0)).alias(name)
      else lit(0.0).alias(name)

    val selected = raw.select(
      col("Attack_type"), col("attack_family"), col("traffic_class"),
      col("proto"), col("service"), col("orig_port_type"), col("resp_port_type"),
      num("flow_duration"), num("fwd_pkts_tot"), num("bwd_pkts_tot"),
      num("total_pkts"), num("fwd_pkt_ratio"),
      num("syn_flag_ratio"), num("rst_flag_ratio")
    )

    // TRANSFORMATION 1: map - parse each Row into a typed Flow object
    val allFlows: RDD[Flow] = selected.rdd.map { r =>
      Flow(
        r.getAs[String]("Attack_type"),
        r.getAs[String]("attack_family"),
        r.getAs[String]("traffic_class"),
        r.getAs[String]("proto"),
        r.getAs[String]("service"),
        r.getAs[String]("orig_port_type"),
        r.getAs[String]("resp_port_type"),
        r.getAs[Double]("flow_duration"),
        r.getAs[Double]("fwd_pkts_tot"),
        r.getAs[Double]("bwd_pkts_tot"),
        r.getAs[Double]("total_pkts"),
        r.getAs[Double]("fwd_pkt_ratio"),
        r.getAs[Double]("syn_flag_ratio"),
        r.getAs[Double]("rst_flag_ratio")
      )
    }

    // TRANSFORMATION 2: filter - keep only flows that carried packets
    val flows = allFlows.filter(_.totalPkts > 0).cache()

    // ACTION 1: count
    header("0. Dataset loaded as an RDD")
    println("Flows loaded:                 " + allFlows.count())
    println("Flows with at least 1 packet: " + flows.count())

    // ACTION 2: first
    println("First flow: " + flows.first())

    // ======================================================================
    // Task 1 - Per-class traffic volume
    // Why: classes differ hugely in size; this gives the size and the average
    // flow shape of each class in real units (seconds, packets).
    // reduceByKey: (count, sum duration, sum packets, max packets)
    // ======================================================================
    header("1. Per-class volume statistics (reduceByKey)")

    val classStats = flows
      .map(f => (f.attackType, (1L, f.duration, f.totalPkts, f.totalPkts)))
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, math.max(a._4, b._4)))
      .cache()

    // ACTION 3: collect (only 12 rows)
    classStats.sortBy(-_._2._1).collect().foreach { case (k, (n, d, p, m)) =>
      println(f"$k%-28s flows=$n%7d  avgDuration=${d / n}%10.4f s  avgPkts=${p / n}%10.2f  maxPkts=$m%10.0f")
    }

    // ======================================================================
    // Task 2 - Behaviour profile per class (derived features)
    // Why: scans and floods are one-directional (no reply) and flag-heavy.
    // aggregateByKey builds (count, one-way flows, sum SYN ratio, sum RST ratio)
    // without creating one object per flow in memory.
    // ======================================================================
    header("2. Behaviour profile per class (aggregateByKey)")

    val profile = flows
      .map(f => (f.attackType, f))
      .aggregateByKey((0L, 0L, 0.0, 0.0))(
        (acc, f) => (
          acc._1 + 1,
          acc._2 + (if (f.fwdRatio >= 1.0) 1L else 0L),
          acc._3 + f.synRatio,
          acc._4 + f.rstRatio),
        (a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, a._4 + b._4)
      )
      // TRANSFORMATION: mapValues - turn the sums into averages / percentages
      .mapValues { case (n, oneWay, syn, rst) => (n, 100.0 * oneWay / n, syn / n, rst / n) }
      .cache()

    profile.sortBy(-_._2._2).collect().foreach { case (k, (n, oneWayPct, syn, rst)) =>
      println(f"$k%-28s one-way flows=$oneWayPct%6.2f%%  avgSynRatio=$syn%.4f  avgRstRatio=$rst%.4f")
    }

    // ======================================================================
    // Task 3 - Join with the attack_family reference table
    // Why: re-uses the Phase 2 reference table to move from 12 classes to
    // 6 attack families, which is the level the report discusses.
    // ======================================================================
    header("3. Family-level statistics (join + reduceByKey)")

    val familyRef = sc.textFile("data/attack_family_csv/*.csv")
      .filter(line => !line.startsWith("Attack_type"))          // drop header
      .map(_.split(","))
      .map(a => (a(0), (a(1), a(2))))                           // (Attack_type, (family, class))

    // TRANSFORMATION: join
    val joined = classStats.join(familyRef)

    val familyStats = joined
      .map { case (_, ((n, d, p, m), (fam, _))) => (fam, (n, d, p, m)) }
      .reduceByKey((a, b) => (a._1 + b._1, a._2 + b._2, a._3 + b._3, math.max(a._4, b._4)))
      .cache()

    familyStats.sortBy(-_._2._1).collect().foreach { case (fam, (n, d, p, m)) =>
      println(f"$fam%-32s flows=$n%7d  avgDuration=${d / n}%10.4f s  avgPkts=${p / n}%10.2f")
    }

    // ACTION 4: reduce - the join must not lose any flow
    val joinedTotal = familyStats.map(_._2._1).reduce(_ + _)
    println("Flows after join: " + joinedTotal + " | flows before: " + flows.count() +
            (if (joinedTotal == flows.count()) "  -> OK, no flows lost" else "  -> MISMATCH"))

    // ======================================================================
    // Task 4 - Packet-count distribution per family (median)
    // Why: the data is heavily skewed, so the mean hides typical behaviour.
    // An exact median needs all values of a group, so groupByKey is justified
    // (only 6 groups, ~118k numbers in total).
    // ======================================================================
    header("4. Median packets per flow by family (groupByKey)")

    val medians = flows
      .map(f => (f.family, f.totalPkts))
      .groupByKey()
      .mapValues { vs =>
        val s = vs.toArray.sorted
        val n = s.length
        if (n % 2 == 1) s(n / 2) else (s(n / 2 - 1) + s(n / 2)) / 2.0
      }

    medians.sortBy(-_._2).collect().foreach { case (fam, med) =>
      println(f"$fam%-32s median packets per flow = $med%.1f")
    }

    // ======================================================================
    // Task 5 - Most common protocol / service / port-type per traffic class
    // Why: shows which categorical values separate normal IoT traffic from
    // attacks (supports the one-hot features used in Phase 5).
    // flatMap: one flow -> four (class, token) pairs
    // ======================================================================
    header("5. Top categorical values by traffic class (flatMap)")

    val tokenCounts = flows
      .flatMap { f =>
        Seq(s"proto=${f.proto}", s"service=${f.service}",
            s"orig_port=${f.origPortType}", s"resp_port=${f.respPortType}")
          .map(t => ((f.trafficClass, t), 1L))
      }
      .reduceByKey(_ + _)
      .cache()

    Seq("Attack", "Normal").foreach { cls =>
      println("Top values for traffic_class = " + cls)
      // ACTION 5: take
      tokenCounts.filter(_._1._1 == cls).sortBy(-_._2).take(8).foreach { case ((_, t), c) =>
        println(f"  $t%-30s $c%8d")
      }
    }

    // ======================================================================
    // Task 6 - How many different services does each family touch?
    // distinct removes repeated (family, service) pairs, countByKey then
    // counts the services per family.
    // ======================================================================
    header("6. Distinct services per family (distinct + countByKey)")

    // ACTION 6: countByKey
    val servicesPerFamily = flows
      .map(f => (f.family, f.service))
      .distinct()
      .countByKey()

    servicesPerFamily.toSeq.sortBy(-_._2).foreach { case (fam, n) =>
      println(f"$fam%-32s distinct services = $n%d")
    }

    // ======================================================================
    // Task 7 - Rare classes (class-imbalance evidence for Phase 5)
    // sortByKey on (count, class) gives the rarest classes; a broadcast set
    // then filters the original RDD to just those flows.
    // ======================================================================
    header("7. Rare classes (sortByKey + broadcast + takeSample)")

    val rarest = classStats
      .map { case (k, (n, _, _, _)) => (n, k) }
      .sortByKey(ascending = true)
      .take(3)
    println("Three rarest classes: " + rarest.map { case (n, k) => s"$k ($n flows)" }.mkString(", "))

    val rareSet = sc.broadcast(rarest.map(_._2).toSet)
    val rareFlows = flows.filter(f => rareSet.value.contains(f.attackType))

    // ACTION 7: takeSample
    println("Random sample of rare-class flows:")
    rareFlows.takeSample(withReplacement = false, 6, 42).foreach { f =>
      println(f"  ${f.attackType}%-28s pkts=${f.totalPkts}%6.0f  fwdRatio=${f.fwdRatio}%.2f  " +
              f"synRatio=${f.synRatio}%.3f  resp_port=${f.respPortType}")
    }

    // ACTION 8: foreach (small RDD, so printing on the executors is fine)
    println("Rare-class average packets per flow:")
    classStats
      .filter { case (k, _) => rareSet.value.contains(k) }
      .foreach { case (k, (n, _, p, _)) => println(f"  $k%-28s avgPkts=${p / n}%.2f") }

    // ======================================================================
    // Task 8 - Heaviest flow and duration skew
    // ======================================================================
    header("8. Heaviest flow and duration distribution (reduce + histogram)")

    val heaviest = flows.reduce((a, b) => if (a.totalPkts >= b.totalPkts) a else b)
    println(f"Flow with the most packets: ${heaviest.attackType} (${heaviest.totalPkts}%.0f packets, " +
            f"${heaviest.duration}%.2f s)")

    // ACTION 9: histogram on log10(1 + duration), 6 equal-width buckets
    val (edges, counts) = flows.map(f => math.log10(1.0 + f.duration)).histogram(6)
    println("Histogram of log10(1 + flow_duration):")
    counts.indices.foreach { i =>
      println(f"  [${edges(i)}%6.2f, ${edges(i + 1)}%6.2f)  ${counts(i)}%8d flows")
    }

    // ======================================================================
    // Save results (ACTION 10: saveAsTextFile)
    // ======================================================================
    header("Saving results")

    saveText(
      classStats.map { case (k, (n, d, p, m)) => s"$k,$n,${d / n},${p / n},$m" },
      "results/rdd_class_stats")
    saveText(
      profile.map { case (k, (n, ow, syn, rst)) => s"$k,$n,$ow,$syn,$rst" },
      "results/rdd_class_profile")
    saveText(
      familyStats.map { case (fam, (n, d, p, m)) => s"$fam,$n,${d / n},${p / n},$m" },
      "results/rdd_family_stats")

    println("Saved: results/rdd_class_stats    (attack_type, flows, avg_duration_s, avg_pkts, max_pkts)")
    println("Saved: results/rdd_class_profile  (attack_type, flows, one_way_pct, avg_syn_ratio, avg_rst_ratio)")
    println("Saved: results/rdd_family_stats   (family, flows, avg_duration_s, avg_pkts, max_pkts)")

    spark.stop()
  }
}
