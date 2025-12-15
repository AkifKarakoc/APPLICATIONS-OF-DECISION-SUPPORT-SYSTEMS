package com.wekaproject;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.Normalize;

public class DataProcessor {
    private Instances originalData;
    private String datasetPath;

    public DataProcessor(String datasetPath) throws Exception {
        this.datasetPath = datasetPath;
        loadData();
    }

    private void loadData() throws Exception {
        DataSource source = new DataSource(datasetPath);
        originalData = source.getDataSet();
        
        // Set class index to last attribute if not set
        if (originalData.classIndex() == -1) {
            originalData.setClassIndex(originalData.numAttributes() - 1);
        }
    }

    public Instances getOriginalData() {
        return new Instances(originalData);
    }

    public int getNumInstances() {
        return originalData.numInstances();
    }

    public int getNumAttributes() {
        return originalData.numAttributes();
    }

    /**
     * Convert all nominal attributes (except class) to binary
     */
    public Instances nominalToBinary() throws Exception {
        Instances data = new Instances(originalData);
        
        NominalToBinary filter = new NominalToBinary();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        
        return data;
    }

    /**
     * Convert all numeric attributes to nominal using discretization
     */
    public Instances numericToNominal() throws Exception {
        Instances data = new Instances(originalData);
        
        Discretize filter = new Discretize();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        
        return data;
    }

    /**
     * Normalize numeric attributes to [0, 1] range
     */
    public Instances normalize(Instances data) throws Exception {
        Normalize filter = new Normalize();
        filter.setInputFormat(data);
        data = Filter.useFilter(data, filter);
        
        return data;
    }

    /**
     * Convert to numeric (NominalToBinary) and normalize
     */
    public Instances toNumericNormalized() throws Exception {
        Instances data = nominalToBinary();
        return normalize(data);
    }

    /**
     * Check if dataset has any nominal attributes (excluding class)
     */
    public boolean hasNominalAttributes() {
        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != originalData.classIndex() && originalData.attribute(i).isNominal()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Check if dataset has any numeric attributes (excluding class)
     */
    public boolean hasNumericAttributes() {
        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != originalData.classIndex() && originalData.attribute(i).isNumeric()) {
                return true;
            }
        }
        return false;
    }

    /**
     * Check if dataset is fully nominal (all attributes except class)
     */
    public boolean isFullyNominal() {
        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != originalData.classIndex() && !originalData.attribute(i).isNominal()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Check if dataset is fully numeric (all attributes except class)
     */
    public boolean isFullyNumeric() {
        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != originalData.classIndex() && !originalData.attribute(i).isNumeric()) {
                return false;
            }
        }
        return true;
    }

    /**
     * Get attribute names (excluding class)
     */
    public java.util.List<String> getAttributeNames() {
        java.util.List<String> names = new java.util.ArrayList<>();
        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != originalData.classIndex()) {
                names.add(originalData.attribute(i).name());
            }
        }
        return names;
    }

    /**
     * Check if attribute at index is numeric
     */
    public boolean isNumericAttribute(int index) {
        return originalData.attribute(index).isNumeric();
    }

    /**
     * Get nominal values for a nominal attribute
     */
    public java.util.List<String> getNominalValues(int index) {
        java.util.List<String> values = new java.util.ArrayList<>();
        if (originalData.attribute(index).isNominal()) {
            java.util.Enumeration<Object> enumeration = originalData.attribute(index).enumerateValues();
            while (enumeration.hasMoreElements()) {
                values.add(enumeration.nextElement().toString());
            }
        }
        return values;
    }

    /**
     * Get attribute index by name (excluding class)
     */
    public int getAttributeIndex(String name) {
        for (int i = 0; i < originalData.numAttributes(); i++) {
            if (i != originalData.classIndex() && originalData.attribute(i).name().equals(name)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Get sample values from the dataset for a specific attribute
     * Returns up to 3 unique sample values from the dataset
     */
    public java.util.List<String> getSampleValues(int attributeIndex, int maxSamples) {
        java.util.List<String> samples = new java.util.ArrayList<>();
        java.util.Set<String> uniqueValues = new java.util.HashSet<>();

        int sampleCount = 0;
        for (int i = 0; i < originalData.numInstances() && sampleCount < maxSamples; i++) {
            if (!originalData.instance(i).isMissing(attributeIndex)) {
                String value;
                if (originalData.attribute(attributeIndex).isNumeric()) {
                    double numValue = originalData.instance(i).value(attributeIndex);
                    // Format numeric values nicely
                    if (numValue == (long) numValue) {
                        value = String.format("%d", (long) numValue);
                    } else {
                        value = String.format("%.2f", numValue);
                    }
                } else {
                    value = originalData.instance(i).stringValue(attributeIndex);
                }

                if (uniqueValues.add(value)) {
                    samples.add(value);
                    sampleCount++;
                }
            }
        }

        return samples;
    }
}
