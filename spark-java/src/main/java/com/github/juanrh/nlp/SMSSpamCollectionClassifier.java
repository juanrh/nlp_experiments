package com.github.juanrh.nlp;

import java.net.URL;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.io.File;
import java.io.IOException;
import java.nio.file.FileSystems;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import scala.Tuple2;

import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.SparkConf;
import org.apache.spark.mllib.classification.NaiveBayes;
import org.apache.spark.mllib.classification.NaiveBayesModel;
import org.apache.spark.mllib.feature.HashingTF;
import org.apache.spark.mllib.regression.LabeledPoint;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.commons.io.FileUtils;

import com.google.common.base.Joiner;
import com.google.common.collect.ImmutableMap;

public class SMSSpamCollectionClassifier {
	private static final Logger LOGGER = LoggerFactory.getLogger(SMSSpamCollectionClassifier.class);
	
	public static void main(String[] args) throws IOException {
		final SparkConf conf = new SparkConf().setAppName("SMSSpamCollectionClassifier")
											  .setMaster("local[*]");
		LOGGER.info("Starting Spark context");
		final JavaSparkContext sc = new JavaSparkContext(conf);
	
		final URL dataPath = SMSSpamCollectionClassifier.class.getResource("/SMSSpamCollection");
		final File outputModelPath = new File(FileSystems.getDefault().getPath("target", "tmp", "SMSSpamCollectionBayesModel").toString());
		final Map<String, Double> labels = ImmutableMap.of("ham", 0.0, "spam", 1.0);
		final JavaRDD<String> data = sc.textFile(dataPath.toString(), 4);
		// assign class and split text in words, also pass to lower case
		// Use TF to vectorize each seq of words
		// Spark's implementation of IDF makes it difficult to use for 
		// feature extraction, but Multinomial Naive Bayes already obtains 
		// a similar effect
		final HashingTF hashingTF = new HashingTF();
		final JavaRDD<LabeledPoint> labelledTF = data.map(line -> {
			String[] parts = line.split("\t");
			Double label = labels.get(parts[0]);
			List<String> words = Stream.of(parts[1].split("\\s+"))
										 .map(String::toLowerCase)
										 .collect(Collectors.toList());;
			return new LabeledPoint(label, hashingTF.transform(words));
		});
		System.out.println(Joiner.on('\n').join(data.take(5)));
		System.out.println(Joiner.on('\n').join(labelledTF.take(5)));
			   
		// Split data into training (60%) and test (40%).
		final JavaRDD<LabeledPoint>[] splits = labelledTF.randomSplit(new double [] {0.6, 0.4}, 11L);
		final JavaRDD<LabeledPoint> train = splits[0];
		final  JavaRDD<LabeledPoint> test = splits[1];
		test.cache();

		final NaiveBayesModel model = NaiveBayes.train(train.rdd(), 1.0, "multinomial");
		
		final JavaPairRDD<Double, Double> predictionAndLabel = test.mapToPair(point -> 
				new Tuple2<>(model.predict(point.features()), point.label())
				);
		final double accuracy = Double.valueOf(predictionAndLabel.filter(x -> x._1.compareTo(x._2) == 0).count()) 
									/ test.count();
		System.out.println(Joiner.on('\n').join(predictionAndLabel.take(5)));
		System.out.println("accuracy: " + accuracy);

		// Save and load model
		System.out.println("saving model to file " + outputModelPath);
		if (outputModelPath.exists()) {
		   FileUtils.deleteDirectory(outputModelPath);
		}
		model.save(sc.sc(), outputModelPath.toString());
		final NaiveBayesModel sameModel = NaiveBayesModel.load(sc.sc(), outputModelPath.toString());
		
		LOGGER.info("Finished computation");
		sc.stop();
	}
}
