package com.wekaproject;

import javax.swing.*;
import java.awt.*;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class DiscoverPanel extends JPanel {
    private ClassificationEngine engine;
    private DataProcessor dataProcessor;
    private List<JComponent> inputComponents;
    private Map<Integer, JComponent> attributeInputMap;
    private JLabel resultLabel;
    private JButton predictButton;

    public DiscoverPanel(ClassificationEngine engine) {
        this.engine = engine;
        this.dataProcessor = engine.getDataProcessor();
        this.inputComponents = new ArrayList<>();
        this.attributeInputMap = new HashMap<>();

        setLayout(new BorderLayout(10, 10));
        setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        initComponents();
    }

    private void initComponents() {
        // Top info panel
        JPanel infoPanel = new JPanel(new BorderLayout());
        infoPanel.setBorder(BorderFactory.createTitledBorder("Prediction Model"));

        JLabel modelInfoLabel = new JLabel(
            "Using best model: " + (engine.isTrained() ? engine.getBestAlgorithmName() : "Not trained yet")
        );
        modelInfoLabel.setFont(new Font("Arial", Font.BOLD, 12));
        infoPanel.add(modelInfoLabel, BorderLayout.CENTER);

        // Form panel with scroll
        JPanel formPanel = new JPanel();
        formPanel.setLayout(new BoxLayout(formPanel, BoxLayout.Y_AXIS));
        formPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        List<String> attributeNames = dataProcessor.getAttributeNames();

        for (int i = 0; i < attributeNames.size(); i++) {
            String attrName = attributeNames.get(i);
            int attrIndex = dataProcessor.getAttributeIndex(attrName);

            // Main row panel containing attribute name, input, and examples
            JPanel rowPanel = new JPanel();
            rowPanel.setLayout(new BoxLayout(rowPanel, BoxLayout.Y_AXIS));
            rowPanel.setBorder(BorderFactory.createEmptyBorder(5, 5, 5, 5));
            rowPanel.setMaximumSize(new Dimension(Integer.MAX_VALUE, 70));

            // First row: label and input
            JPanel inputRow = new JPanel(new BorderLayout(10, 5));

            // Label
            JLabel label = new JLabel(attrName + ":");
            label.setPreferredSize(new Dimension(150, 25));
            inputRow.add(label, BorderLayout.WEST);

            // Input component
            JComponent inputComponent;

            if (dataProcessor.isNumericAttribute(attrIndex)) {
                // Numeric attribute - use text field
                JTextField textField = new JTextField();
                textField.setPreferredSize(new Dimension(200, 25));
                inputComponent = textField;
            } else {
                // Nominal attribute - use combo box
                List<String> nominalValues = dataProcessor.getNominalValues(attrIndex);
                JComboBox<String> comboBox = new JComboBox<>(nominalValues.toArray(new String[0]));
                comboBox.setPreferredSize(new Dimension(200, 25));
                inputComponent = comboBox;
            }

            inputRow.add(inputComponent, BorderLayout.CENTER);

            // Second row: example values
            List<String> sampleValues = dataProcessor.getSampleValues(attrIndex, 3);
            if (!sampleValues.isEmpty()) {
                JPanel exampleRow = new JPanel(new BorderLayout());
                JLabel exampleLabel = new JLabel("   Examples: " + String.join(", ", sampleValues));
                exampleLabel.setFont(new Font("Arial", Font.ITALIC, 10));
                exampleLabel.setForeground(new Color(100, 100, 100));
                exampleRow.add(exampleLabel, BorderLayout.WEST);

                rowPanel.add(inputRow);
                rowPanel.add(exampleRow);
            } else {
                rowPanel.add(inputRow);
            }

            formPanel.add(rowPanel);

            inputComponents.add(inputComponent);
            attributeInputMap.put(attrIndex, inputComponent);
        }

        JScrollPane scrollPane = new JScrollPane(formPanel);
        scrollPane.setBorder(BorderFactory.createTitledBorder("Enter Attribute Values"));
        scrollPane.setVerticalScrollBarPolicy(JScrollPane.VERTICAL_SCROLLBAR_AS_NEEDED);
        scrollPane.setHorizontalScrollBarPolicy(JScrollPane.HORIZONTAL_SCROLLBAR_NEVER);

        // Bottom panel with button and result
        JPanel bottomPanel = new JPanel(new BorderLayout(10, 10));
        bottomPanel.setBorder(BorderFactory.createEmptyBorder(10, 10, 10, 10));

        predictButton = new JButton("Predict Class");
        predictButton.setFont(new Font("Arial", Font.BOLD, 14));
        predictButton.setPreferredSize(new Dimension(0, 40));
        predictButton.addActionListener(e -> performPrediction());
        predictButton.setEnabled(engine.isTrained());

        resultLabel = new JLabel("Prediction result will appear here", SwingConstants.CENTER);
        resultLabel.setFont(new Font("Arial", Font.BOLD, 16));
        resultLabel.setForeground(new Color(0, 100, 200));
        resultLabel.setBorder(BorderFactory.createCompoundBorder(
            BorderFactory.createLineBorder(Color.LIGHT_GRAY, 1),
            BorderFactory.createEmptyBorder(20, 20, 20, 20)
        ));
        resultLabel.setOpaque(true);
        resultLabel.setBackground(new Color(240, 248, 255));

        bottomPanel.add(predictButton, BorderLayout.NORTH);
        bottomPanel.add(resultLabel, BorderLayout.CENTER);

        // Add all to main panel
        add(infoPanel, BorderLayout.NORTH);
        add(scrollPane, BorderLayout.CENTER);
        add(bottomPanel, BorderLayout.SOUTH);
    }

    private void performPrediction() {
        if (!engine.isTrained()) {
            JOptionPane.showMessageDialog(
                this,
                "Model is not trained yet. Please run classification first.",
                "Not Trained",
                JOptionPane.WARNING_MESSAGE
            );
            return;
        }

        try {
            // Collect and validate input values
            List<String> attributeNames = dataProcessor.getAttributeNames();
            double[] attributeValues = new double[attributeNames.size()];

            for (int i = 0; i < attributeNames.size(); i++) {
                String attrName = attributeNames.get(i);
                int attrIndex = dataProcessor.getAttributeIndex(attrName);
                JComponent component = attributeInputMap.get(attrIndex);

                if (dataProcessor.isNumericAttribute(attrIndex)) {
                    // Numeric attribute
                    JTextField textField = (JTextField) component;
                    String text = textField.getText().trim();

                    if (text.isEmpty()) {
                        JOptionPane.showMessageDialog(
                            this,
                            "Please enter a value for attribute: " + attrName,
                            "Missing Value",
                            JOptionPane.WARNING_MESSAGE
                        );
                        textField.requestFocus();
                        return;
                    }

                    try {
                        attributeValues[attrIndex] = Double.parseDouble(text);
                    } catch (NumberFormatException e) {
                        JOptionPane.showMessageDialog(
                            this,
                            "Invalid numeric value for attribute: " + attrName + "\nPlease enter a valid number.",
                            "Invalid Input",
                            JOptionPane.ERROR_MESSAGE
                        );
                        textField.requestFocus();
                        return;
                    }
                } else {
                    // Nominal attribute
                    JComboBox<String> comboBox = (JComboBox<String>) component;
                    String selectedValue = (String) comboBox.getSelectedItem();

                    if (selectedValue == null) {
                        JOptionPane.showMessageDialog(
                            this,
                            "Please select a value for attribute: " + attrName,
                            "Missing Value",
                            JOptionPane.WARNING_MESSAGE
                        );
                        return;
                    }

                    // Get the index of the selected nominal value
                    List<String> nominalValues = dataProcessor.getNominalValues(attrIndex);
                    attributeValues[attrIndex] = nominalValues.indexOf(selectedValue);
                }
            }

            // Perform prediction
            String predictedClass = engine.predictClass(attributeValues);

            // Display result
            resultLabel.setText("Predicted Class: " + predictedClass);
            resultLabel.setForeground(new Color(0, 128, 0));

        } catch (Exception e) {
            JOptionPane.showMessageDialog(
                this,
                "Error during prediction: " + e.getMessage(),
                "Prediction Error",
                JOptionPane.ERROR_MESSAGE
            );
            resultLabel.setText("Prediction failed");
            resultLabel.setForeground(Color.RED);
        }
    }
}

