package com.intel.analytics.zoo.apps.streaming.textclassification;

import com.intel.analytics.zoo.pipeline.inference.JTensor;

import java.io.File;
import java.util.HashMap;
import java.util.Map;

public class TextProcessor {
    private int stopWordsCount;
    private int sequenceLength;
    private Map<String, Integer> wordToIndexMap;

    public TextProcessor(int stopWordsCount, int sequenceLength, String embeddingFilePath) {
        this.stopWordsCount = stopWordsCount;
        this.sequenceLength = sequenceLength;
        this.wordToIndexMap = loadWordToIndexMap(new File(embeddingFilePath));
    }

    public JTensor preprocess(String text) {
        return new JTensor();
    }

    private Map<String, Integer> loadWordToIndexMap(File file) {
        return new HashMap<String, Integer>();
    }
}
