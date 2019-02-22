#
# Copyright 2018 Analytics Zoo Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import sys
import datetime as dt
from optparse import OptionParser

from bigdl.optim.optimizer import Adagrad
from zoo.common.nncontext import init_nncontext
from zoo.feature.text import TextSet
from zoo.models.textclassification import TextClassifier
from pyspark import SparkContext
from pyspark.streaming import StreamingContext


if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("--embedding_path", dest="embedding_path")
    parser.add_option("--class_num", dest="class_num", default="20")
    parser.add_option("--partition_num", dest="partition_num", default="4")
    parser.add_option("--token_length", dest="token_length", default="200")
    parser.add_option("--sequence_length", dest="sequence_length", default="500")
    parser.add_option("--max_words_num", dest="max_words_num", default="5000")
    parser.add_option("--encoder", dest="encoder", default="cnn")
    parser.add_option("--encoder_output_dim", dest="encoder_output_dim", default="256")
    parser.add_option("--training_split", dest="training_split", default="0.8")
    parser.add_option("-b", "--batch_size", dest="batch_size", default="128")
    parser.add_option("-e", "--nb_epoch", dest="nb_epoch", default="20")
    parser.add_option("-l", "--learning_rate", dest="learning_rate", default="0.01")
    parser.add_option("--log_dir", dest="log_dir", default="/tmp/.analytics-zoo")
    parser.add_option("-m", "--model", dest="model")

    (options, args) = parser.parse_args(sys.argv)
    sc = SparkContext("local[2]", "NetworkTextClassification")
    ssc = StreamingContext(sc, 1)

    lines = ssc.socketTextStream("localhost", 9999)
    # Load lines from network streaming into text_set
    ## TODO Read TextSet from stream
    print("Processing text dataset...")
    transformed = text_set.tokenize().normalize()\
        .word2idx(remove_topN=10, max_words_num=int(options.max_words_num))\
        .shape_sequence(len=int(options.sequence_length)).generate_sample()
    if options.model:
        model = TextClassifier.load_model(options.model)
    else:
        print("ERROR: Model not given!")
        exit()
    predict_set = model.predict(transformed, batch_per_thread=int(options.partition_num))
    # Get the top 5 prediction probability distributions
    predicts = predict_set.get_predicts().take(5)
    print("Probability distributions of the first five texts in the validation set:")
    for predict in predicts:
        print(predict)
    ssc.start()
    ssc.awaitTermination()
