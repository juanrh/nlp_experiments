import com.typesafe.sbteclipse.plugin.EclipsePlugin.EclipseKeys._

// -Xsource:2.10 -Ymacro-expand:none

name := "nlp_experiments_spark"

organization := "com.github.juanrh"

version := "0.0.1" 

scalaVersion := "2.10.6"

crossScalaVersions  := Seq("2.10.6")

licenses += ("Apache-2.0", url("http://www.apache.org/licenses/LICENSE-2.0"))

lazy val sparkVersion = "1.6.0"

// Use `sbt doc` to generate scaladoc, more on chapter 14.8 of "Scala Cookbook"

// show all the warnings: http://stackoverflow.com/questions/9415962/how-to-see-all-the-warnings-in-sbt-0-11
scalacOptions ++= Seq("-feature", "-unchecked", "-deprecation")

// if parallel test execution is not disabled and several test suites using
// SparkContext (even through SharedSparkContext) are running then tests fail randomly
parallelExecution := false

// Could be interesting at some point
// resourceDirectory in Compile := baseDirectory.value / "main/resources"
// resourceDirectory in Test := baseDirectory.value / "main/resources"

// Configure sbt to add the resources path to the eclipse project http://stackoverflow.com/questions/14060131/access-configuration-resources-in-scala-ide
// This is critical so log4j.properties is found by eclipse
EclipseKeys.createSrc := EclipseCreateSrc.Default + EclipseCreateSrc.Resource

// Spark 
libraryDependencies += "org.apache.spark" %% "spark-core" % sparkVersion

libraryDependencies += "org.apache.spark" %% "spark-mllib" % sparkVersion

// libraryDependencies += "org.apache.spark" %% "spark-sql" % sparkVersion


// note this is discontinued for scala 2.11, which uses https://github.com/typesafehub/scala-logging#contribution-policy
libraryDependencies += "com.typesafe" % "scalalogging-log4j_2.10" % "1.1.0"

libraryDependencies += "com.typesafe" %% "scalalogging-slf4j" % "1.1.0"
