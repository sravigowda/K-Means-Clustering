import org.apache.log4j.Level;

import org.apache.log4j.Logger;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;
import org.apache.spark.sql.expressions.Window;
import org.apache.spark.sql.expressions.WindowSpec;
import org.apache.spark.ml.feature.Bucketizer;
import org.apache.spark.ml.feature.Imputer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.rdd.RDD;
import org.apache.spark.ml.feature.ImputerModel;
import org.apache.spark.ml.feature.MaxAbsScaler;
import org.apache.spark.ml.feature.MaxAbsScalerModel;
import org.apache.spark.ml.feature.Normalizer;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.RelationalGroupedDataset;
import org.apache.spark.sql.types.DataTypes;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SQLContext;

import static org.apache.spark.sql.functions.col;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.ml.evaluation.*;
import com.github.fommil.netlib.*;
import org.apache.spark.sql.DataFrameNaFunctions;
import org.apache.spark.ml.clustering.KMeansModel;
import org.apache.spark.ml.clustering.KMeans;
import org.apache.spark.ml.evaluation.ClusteringEvaluator;
import org.apache.spark.ml.linalg.Vector;
import org.apache.spark.sql.DataFrameReader;
import org.apache.spark.sql.DataFrameWriter;

import java.util.Arrays;
import java.util.List;

import org.apache.spark.sql.types.StructField;
import org.apache.spark.sql.types.StructType;

import scala.Tuple2;
import scala.collection.Seq;

import org.apache.hadoop.classification.InterfaceStability.Stable;
import static org.apache.spark.sql.functions.*;

public class AnalyticsClustering {
	
	public static void main (String[] args) {
		
		System.out.println("This is somthing idiotic");
		
		Logger.getLogger("org").setLevel(Level.ERROR);
		
		SparkSession sparkSession = SparkSession.builder().appName("AnalyticsClustering").master("local[*]").getOrCreate();
		
		
		Dataset<Row> ClickStreamData = sparkSession.read().option("header", false).option("inferschema", false)
				.csv("/Users/ravig/Documents/Upgrad/Complete/Course6/Project/sample100mb.csv")
				.toDF("UserID","epoch_seconds","SongID","Date");
		
		Dataset<Row> SongsMetaData = sparkSession.read().option("header", false).option("inferschema", false)
				.csv("/Users/ravig/Documents/Upgrad/Complete/Course6/Project/newmetadata/part-m-*")
				.toDF("SongID","ArtistID");
		
		Dataset<Row> SongsMetaData_nonull = SongsMetaData.na().drop("any");
		//SongsMetaData_nonull.show(300);
		
	
			
		
		Dataset<Row> ClickStreamData_new = ClickStreamData.select(col("UserID"),col("epoch_seconds").cast("Integer"),col("SongID"),col("Date"));
		Dataset<Row> ClickStreamData_count = ClickStreamData_new.withColumn("Count", functions.lit(1));
		
		ClickStreamData_count.show();
		ClickStreamData_count.printSchema();
		
		
		Dataset<Row> UniqueSongHitPerDay = ClickStreamData_count.groupBy("UserID","SongID","Date").sum("Count")
				.withColumnRenamed("sum(Count)", "ListenedCount");
		
		System.out.println("Printing ClickStream Data with UniqueSongHitPerDay");
		
		
		
		//UniqueSongHitPerDay.orderBy("ListenedCount").show();
		//UniqueSongHitPerDay.orderBy(org.apache.spark.sql.functions.col("ListenedCount").desc()).show();
		//UniqueSongHitPerDay.printSchema();
		//UniqueSongHitPerDay.show();
		
		System.out.println("Printing ClickStream Data with TotalSongsPerDay");
		
		Dataset<Row> User_TotalSongsPerDay = UniqueSongHitPerDay
				.groupBy("UserID","Date").sum("ListenedCount")
				.withColumnRenamed("sum(ListenedCount)", "TotalSongsPerDay");
				
				//User_TotalSongsPerDay.show();
		
		
		System.out.println("Printing ClickStream Data with Unique Songs hit and TotalSongsPerDay");
		
		
	
		Dataset<Row> User_Unique_songs_Total_songs_count = UniqueSongHitPerDay.join(User_TotalSongsPerDay, 
				User_TotalSongsPerDay.col("UserID").equalTo(UniqueSongHitPerDay.col("UserID"))
				.and(User_TotalSongsPerDay.col("Date").equalTo(UniqueSongHitPerDay.col("Date"))))
				.drop(User_TotalSongsPerDay.col("Date")).drop(UniqueSongHitPerDay.col("UserID"));
				
		
		//User_Unique_songs_Total_songs_count.show();
		
		System.out.println("Printing TotalSongsPerUser Songs ");
		
		Dataset<Row> TotalSongsPerUser = ClickStreamData_count.groupBy("UserID").sum("Count")
				.withColumnRenamed("sum(Count)","TotalSongsPlayedoverall");
		//TotalSongsPerUser.orderBy("TotalSongsPlayedoverall").show();
		//TotalSongsPerUser.orderBy(org.apache.spark.sql.functions.col("TotalSongsPlayedoverall").desc()).show();
		//TotalSongsPerUser.printSchema();
		
		
		Dataset<Row> User_Unique_songs_Total_songs_count_overall = User_Unique_songs_Total_songs_count.join(TotalSongsPerUser,"UserID");
		
		//User_Unique_songs_Total_songs_count_overall.orderBy(org.apache.spark.sql.functions.col("TotalSongsPlayedoverall").desc()).show();
		
		
		System.out.println("User_Unique_songs_Total_songs_count with Count with ArtistID");
		
		Dataset<Row> User_Unique_songs_Total_songs_count_artist = User_Unique_songs_Total_songs_count_overall.join(SongsMetaData_nonull, "SongID");
		//User_Unique_songs_Total_songs_count_artist.show();
		//User_Unique_songs_Total_songs_count_artist.printSchema();
		Dataset<Row> User_Unique_songs_Total_songs_count_artist_clean = User_Unique_songs_Total_songs_count_artist.na().drop("any");
		User_Unique_songs_Total_songs_count_artist_clean.show(100);

		VectorAssembler assembler = new VectorAssembler()
				  .setInputCols(new String[] {"ListenedCount", "TotalSongsPerDay", "TotalSongsPlayedoverall"}).setOutputCol("features");
	
		Dataset<Row> datasetRfm = assembler.transform(User_Unique_songs_Total_songs_count_artist_clean);
		datasetRfm.show();
		
		KMeans kmeans = new KMeans().setK(3);
		KMeansModel model = kmeans.fit(datasetRfm);
		
		Dataset<Row> predictions = model.transform(datasetRfm);
		predictions.show(200);
		
		// Evaluate clustering by computing Silhouette score
				ClusteringEvaluator evaluator = new ClusteringEvaluator();

				double silhouette = evaluator.evaluate(predictions);
				System.out.println("Silhouette with squared euclidean distance = " + silhouette);

				// Shows the result
				Vector[] centers = model.clusterCenters();
				System.out.println("Cluster Centers: ");
				for (Vector center : centers) {	
					System.out.println(center);
				}
				
		
		//Dataset<Row> SongsFrequencyOverall = ClickStreamMetaData_clean.groupBy("UserID","Date").sum("ListenedCount")
			//	.withColumnRenamed("sum(ListenedCount)", "TotalSongsPerDay");
		
		//SongsFrequencyOverall.orderBy("TotalSongsPerDay").show();
		
		
		
		
		//ClickStreamMetaData_nonull.orderBy("ListenedCount").show(300);
		//ClickStreamMetaData_nonull.orderBy(org.apache.spark.sql.functions.col("ListenedCount").desc()).show();
		//ClickStreamMetaData_nonull.printSchema();
		
		
		
		
		//Dataset<Row> SongsFrequencyOverall = ClickStreamData_count.groupBy("UserID","SongID").sum("Count")
		//	.withColumnRenamed("sum(Count)", "SongsHitOverall");
		
		

		
		
		//SongsFrequencyOverall.orderBy("SongsHitOverall").show();
		//SongsFrequencyOverall.orderBy(org.apache.spark.sql.functions.col("SongsHitOverall").desc()).show();
		//SongsFrequencyOverall.printSchema();
		
		//Dataset<Row> ClickStreamMetaData_clean_overall = 
		//		ClickStreamMetaData_clean.join(SongsFrequencyOverall,
		//				(ClickStreamMetaData_clean.col("UserID").equalTo(SongsFrequencyOverall.col("UserID"))),"left");
						
		//Dataset<Row> ClickStreamMetaData_clean_overall_both = ClickStreamMetaData_clean.join(SongsFrequencyOverall)
		//		.where (ClickStreamMetaData_clean.col("UserID").equalTo(SongsFrequencyOverall.col("UserID"))
		//				.and (ClickStreamMetaData_clean.col("SongID").equalTo(SongsFrequencyOverall.col("SongID"))));
		
		//ClickStreamMetaData_clean_overall_both.show();
		
				
				
		/*

		Dataset<Row> SongsPerDate = ClickStreamData_count.groupBy("UserID","Date").sum("Count")
				.withColumnRenamed("sum(Count)","TotalPlayedperDay");
		SongsPerDate.orderBy("TotalPlayedperDay").show();
		SongsPerDate.orderBy(org.apache.spark.sql.functions.col("TotalPlayedperDay").desc()).show();
		SongsPerDate.printSchema();
		
				
		
		System.out.println("Uniques Users in ListenedCount per User Table--> " +SumValue.count());
		System.out.println("Uniques Users in TotalPlayed per User Table--> " +SongsPerDate.count());
		*/
	}

}
