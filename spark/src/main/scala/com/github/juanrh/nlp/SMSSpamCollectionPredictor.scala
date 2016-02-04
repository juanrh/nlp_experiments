package com.github.juanrh.nlp

import scala.io.Source
import java.io.File
import java.nio.file.FileSystems

import org.apache.spark.SparkConf
import org.apache.spark.SparkContext
// Although the NaiveBayesModel.load has no spark context argument this
// throws "org.apache.spark.SparkException: A master URL must be set in your configuration"
// if no master is set
// import org.apache.spark.ml.classification.NaiveBayesModel
import org.apache.spark.mllib.classification.NaiveBayesModel
import org.apache.spark.mllib.feature.HashingTF

/*
16/02/03 21:03:01 WARN NativeCodeLoader: Unable to load native-hadoop library for your platform... using builtin-java classes where applicable
reading model from target\tmp\SMSSpamCollectionBayesModel
16/02/03 21:03:04 WARN : Your hostname, juanyDell resolves to a loopback/non-reachable address: fe80:0:0:0:1021:43de:6c5d:1897%eth6, but we couldn't find any external IP address!
SLF4J: Failed to load class "org.slf4j.impl.StaticLoggerBinder".
SLF4J: Defaulting to no-operation (NOP) logger implementation
SLF4J: See http://www.slf4j.org/codes.html#StaticLoggerBinder for further details.
16/02/03 21:03:06 WARN ParquetRecordReader: Can not initialize counter due to context is not a instance of TaskInputOutputContext, but is org.apache.hadoop.mapreduce.task.TaskAttemptContextImpl
reading data from src\main\resources\SMSSpamCollection
computing predictions
16/02/03 21:03:08 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeSystemBLAS
"predicted "ham" for line [ham	Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...]"
"predicted "ham" for line [ham	Ok lar... Joking wif u oni...]"
"predicted "spam" for line [spam	Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's]"
"predicted "ham" for line [ham	U dun say so early hor... U c already then say...]"
"predicted "ham" for line [ham	Nah I don't think he goes to usf, he lives around here though]"
"predicted "ham" for line [spam	FreeMsg Hey there darling it's been 3 week's now and no word back! I'd like some fun you up for it still? Tb ok! XxX std chgs to send, £1.50 to rcv]"
"predicted "ham" for line [ham	Even my brother is not like to speak with me. They treat me like aids patent.]"
"predicted "ham" for line [ham	As per your request 'Melle Melle (Oru Minnaminunginte Nurungu Vettam)' has been set as your callertune for all Callers. Press *9 to copy your friends Callertune]"
"predicted "spam" for line [spam	WINNER!! As a valued network customer you have been selected to receivea £900 prize reward! To claim call 09061701461. Claim code KL341. Valid 12 hours only.]"
16/02/03 21:03:08 WARN BLAS: Failed to load implementation from: com.github.fommil.netlib.NativeRefBLAS
"predicted "spam" for line [spam	Had your mobile 11 months or more? U R entitled to Update to the latest colour mobiles with camera for Free! Call The Mobile Update Co FREE on 08002986030]"
bye
 * */

object SMSSpamCollectionPredictor extends App {
  // MLlib and Spark ML need a spark context to work, but that can 
  // be local and minimal if we are only interested in scoring
  val conf = new SparkConf().setAppName("SMSSpamCollectionPredictor")
                            .setMaster("local[2]")
  val sc = new SparkContext(conf)
  
  val modelPath = FileSystems.getDefault().getPath("target", "tmp", "SMSSpamCollectionBayesModel").toString 
  println(s"reading model from $modelPath")
  val model = NaiveBayesModel.load(sc, modelPath)
  // Note the hashing term filter doesn't need to be
  // serialized, because it has no state, we only need to 
  // use the same value for numFeatures. Anyway there could 
  // be problems on updates of MLlib, a better solution would 
  // be serializing with SparkML, which would also introduce pipelines
  // which is cleaner
  val hashingTF = new HashingTF()
  val labels = List("ham", "spam")
  
  val dataPath = FileSystems.getDefault().getPath("src", "main", "resources", "/SMSSpamCollection").toString
  println(s"reading data from $dataPath")
  val data = Source.fromFile(dataPath)
  
  println("computing predictions")
  for (line <- data.getLines.take(10)) {
    val words : Seq[String] = line.dropWhile{ _ !=  '\t' }.drop(1)
                                  .split("\\s+").map(_.toLowerCase)
    val prediction = labels(model.predict(hashingTF.transform(words)).toInt)
    println(s""""predicted "${prediction}" for line [${line}]"""")
  }
  println("bye")
  
  sc.stop()
}