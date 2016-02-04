package com.github.juanrh.nlp

import java.io.File
import java.nio.file.FileSystems

import org.apache.commons.io.FileUtils
import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
import org.apache.spark.mllib.classification.NaiveBayes
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.feature.HashingTF
import org.apache.spark.mllib.regression.LabeledPoint
/**
 * Classifier for the UCI "SMS Spam Collection Data Set" https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection  
 * 
 * Feature extraction using Spark's TF http://spark.apache.org/docs/latest/mllib-feature-extraction.html, 
 * see also http://help.mortardata.com/technologies/spark/train_a_machine_learning_model
 * Classifier Spark's using multinomial naive bayes https://spark.apache.org/docs/latest/mllib-naive-bayes.html
 **/

/*
 * Output:
 * 

16/02/01 22:59:43 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
16/02/01 22:59:46 WARN : Your hostname, juanyDell resolves to a loopback/non-reachable address: fe80:0:0:0:1021:43de:6c5d:1897%eth6, but we couldn't find any external IP address!
ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...
ham	Ok lar... Joking wif u oni...
spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's
ham	U dun say so early hor... U c already then say...
ham	Nah I don't think he goes to usf, he lives around here though
(0.0,(1048576,[101,110,3304,3365,3445,52877,72594,100881,101065,102540,226916,270252,294750,520321,664374,696044,769694,810140,853828,956573],[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
(0.0,(1048576,[117,3548,7772,117716,446385,553924],[1.0,1.0,1.0,1.0,1.0,1.0]))
(1.0,(1048576,[50,97,3259,3365,3707,5740,98878,107877,115312,117724,160483,198770,410925,416566,467915,491744,505697,640481,754542,801716,953287,962319,967030,975569],[1.0,1.0,2.0,1.0,3.0,1.0,1.0,1.0,1.0,1.0,1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
(0.0,(1048576,[99,117,3676,99837,113643,413213,570947,780963,857955,897112],[1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
(0.0,(1048576,[105,3325,3707,32662,50637,53232,108821,224519,230642,345520,453540,1043899],[1.0,2.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0]))
16/02/01 22:59:47 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
16/02/01 22:59:47 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
(1.0,1.0)
(0.0,0.0)
(1.0,1.0)
(0.0,0.0)
(1.0,1.0)
accuracy: 93.493 %
saving model to file target\tmp\SMSSpamCollectionBayesModel
16/02/01 22:59:49 WARN TaskSetManager: Stage 8 contains a task of very large size (16386 KB). The maximum recommended task size is 100 KB.
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
16/02/01 22:59:50 WARN ParquetRecordReader: Can not initialize counter due to context is not a instance of TaskInputOutputContext, but is org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl

* */

 /*
 * TODO: implemement as SparkML pipeline https://spark.apache.org/docs/latest/ml-guide.html#algorithm-guides
 * TODO: build RoC curve for TPR and FPR
 * TODO: perform https://en.wikipedia.org/wiki/Text_segmentation with https://github.com/scalanlp/chalk instead
 * of the basic parsing so far (e.g. support stop words)
 * */
object SMSSpamCollectionClassifier extends App {
  val conf = new SparkConf().setAppName("SMSSpamCollectionClassifier")
                            .setMaster("local[*]")
  val sc = new SparkContext(conf)
 
  val dataPath = getClass.getResource("/SMSSpamCollection")
  val outputModelPath = new File(FileSystems.getDefault().getPath("target", "tmp", "SMSSpamCollectionBayesModel").toString())
  val labels = Map("ham" -> 0.0, "spam" -> 1.0)
  
  val data = sc.textFile(dataPath.toString, minPartitions=4)
  // assign class and split text in words, also pass to lower case
  // Use TF to vectorize each seq of words
  // Spark's implementation of IDF makes it difficult to use for 
  // feature extraction, but Multinomial Naive Bayes already obtains 
  // a similar effect
  val hashingTF = new HashingTF()
  val labelledTF =  data.map {line =>
    val parts = line.split('\t')
    val label = labels(parts(0))
    val words : Seq[String] = parts(1).split("\\s+").map(_.toLowerCase)
    LabeledPoint(label, hashingTF.transform(words))
  }
  println(data.take(5).mkString("\n"))   
  println(labelledTF.take(5).mkString("\n"))
   
  // Split data into training (60%) and test (40%).
  val splits = labelledTF.randomSplit(Array(0.6, 0.4), seed = 11L)
  val (train, test) = (splits(0), splits(1))
  test.cache()
  
  val model = NaiveBayes.train(train, lambda = 1.0, modelType = "multinomial")

  val predictionAndLabel = test.map(p => (model.predict(p.features), p.label))
  val accuracy = 1.0 * predictionAndLabel.filter(x => x._1 == x._2).count() / test.count()
  println(predictionAndLabel.take(5).mkString("\n"))
  println(f"accuracy: ${accuracy * 100}%2.3f %%")

  // Save and load model
  println(s"saving model to file $outputModelPath")
  if (outputModelPath.exists()) {
    FileUtils.deleteDirectory(outputModelPath)
  }
  model.save(sc, outputModelPath.toString)
  val sameModel = NaiveBayesModel.load(sc, outputModelPath.toString())
   
  sc.stop()  
}