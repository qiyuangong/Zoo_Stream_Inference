package com.intel.analytics.zoo.apps.streaming

import java.nio.file.Paths

import com.intel.analytics.zoo.common.{NNContext, Utils}
import com.intel.analytics.zoo.feature.image.ImageSet
import com.intel.analytics.zoo.models.image.objectdetection.{ObjectDetector, Visualizer}
import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, Path}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.streaming.{Seconds, StreamingContext}
import org.opencv.imgcodecs.Imgcodecs
import scopt.OptionParser

object StreamingObjectDetection {
  Logger.getLogger("org").setLevel(Level.ERROR)
  Logger.getLogger("akka").setLevel(Level.ERROR)
  Logger.getLogger("breeze").setLevel(Level.ERROR)
  Logger.getLogger("com.intel.analytics.zoo").setLevel(Level.INFO)

  val logger = Logger.getLogger(getClass)

  case class PredictParam(image: String = "file:///tmp/zoo/streaming",
                          outputFolder: String = "data/demo",
                          modelPath: String = "",
                          nPartition: Int = 1)

  val parser = new OptionParser[PredictParam]("Analytics Zoo Object Detection Demo") {
    head("Analytics Zoo Object Detection Demo")
    opt[String]('i', "image")
      .text("where you put the demo image data, can be image folder or image path")
      .action((x, c) => c.copy(image = x))
    opt[String]('o', "output")
      .text("where you put the output data")
      .action((x, c) => c.copy(outputFolder = x))
      .required()
    opt[String]("modelPath")
      .text("Analytics Zoo model path")
      .action((x, c) => c.copy(modelPath = x))
    opt[Int]('p', "partition")
      .text("number of partitions")
      .action((x, c) => c.copy(nPartition = x))
  }

  def main(args: Array[String]): Unit = {
    parser.parse(args, PredictParam()).foreach { params =>
      val sc = NNContext.initNNContext("Streaming Object Detection")
      val ssc = new StreamingContext(sc, Seconds(2))

      // Load pre-trained model
      val model = ObjectDetector.loadModel[Float](params.modelPath)
      // Read image stream from HDFS

//      val fStream = ssc.fileStream(params.image)
//      val data = ImageSet.read(params.image, sc, params.nPartition,
//        imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)

      val lines = ssc.textFileStream(params.image)
      println(lines.toString)
      lines.foreachRDD { batchPath =>
        // Read image files and load to RDD
        println("batchPath partition " + batchPath.getNumPartitions)
        println("batchPath count " + batchPath.count())
        if (!batchPath.isEmpty()) {
          println(batchPath.top(1).apply(0))
          batchPath.foreach { path =>
            println("image path " + path)
            predictImg(path, model)
          }
        }
      }
      ssc.start()
      ssc.awaitTermination()
      logger.info(s"labeled images are saved to ${params.outputFolder}")
    }
  }

  def predictImg(path: String, model: ObjectDetector[Float]): Unit = {
    val fspath = new Path(path)
    val fs = FileSystem.get(fspath.toUri, new Configuration())
    val data = if (path.contains("hdfs")) {
      // Read HDFS image
      val instream= fs.open(fspath)
      val data = new Array[Byte](fs.getFileStatus(new Path(path))
        .getLen.toInt)
      instream.readFully(data)
      instream.close()
      ImageSet.array(Array.apply(data),
        imageCodec =Imgcodecs.CV_LOAD_IMAGE_COLOR)
    } else {
      // Read local image
      ImageSet.read(path, null, 1,
        imageCodec = Imgcodecs.CV_LOAD_IMAGE_COLOR)
    }
    // Read local

    val output = model.predictImageSet(data)
    // Print result
    val visualizer = Visualizer(model.getConfig.labelMap, encoding = "jpg")
//    val visualized = visualizer(output).toDistributed()
//    val result = visualized.rdd.map(imageFeature =>
//    (imageFeature.uri(), imageFeature[Array[Byte]](Visualizer.visualized))).collect()
//    result.foreach(x => {
//      Utils.saveBytes(x._2, getOutPath("output", x._1, "jpg"), true)
//    })
    val visualized = visualizer(output).toLocal()
    val result = visualized.array
    result.foreach(x => {
      if (path.contains("hdfs")) {
        //Save to HDFS dir
        val outstream = fs.create(
          new Path("detection_" + x.uri() + ".jpg"),
          true)
        outstream.write(x[Array[Byte]](Visualizer.visualized))
        outstream.close()
      } else {
        // Save to local dir
        Utils.saveBytes(x[Array[Byte]](Visualizer.visualized),
          getOutPath("output", x.uri(), "jpg"), true)
      }
    })
  }

  private def getOutPath(outPath: String, uri: String, encoding: String): String = {
    Paths.get(outPath,
      s"detection_${ uri.substring(uri.lastIndexOf("/") + 1,
        uri.lastIndexOf(".")) }.${encoding}").toString
  }
}
