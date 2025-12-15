package com.wekaproject;

import javax.swing.*;
import javax.swing.table.DefaultTableModel;
import java.awt.*;
import java.io.File;
import java.util.List;

public class Main extends JFrame {
    private JTextField datasetPathField;
    private JLabel datasetInfoLabel;
    private JButton browseButton;
    private JButton startButton;
    private JProgressBar progressBar;
    private JTable resultsTable;
    private DefaultTableModel tableModel;
    private JLabel bestAlgorithmLabel;
    private File selectedDataset;
    private JTabbedPane tabbedPane;
    private ClassificationEngine classificationEngine;

    public Main() {
        setTitle("WEKA Classifier Comparison");
        setSize(800, 600);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        
        initComponents();
        layoutComponents();
    }

    private void initComponents() {
        // Dataset selection
        datasetPathField = new JTextField();
        datasetPathField.setEditable(false);
        browseButton = new JButton("Browse...");
        browseButton.addActionListener(e -> selectDataset());
        
        datasetInfoLabel = new JLabel("No dataset selected");
        
        // Start button
        startButton = new JButton("Start Classification");
        startButton.setEnabled(false);
        startButton.addActionListener(e -> startClassification());
        
        // Progress bar
        progressBar = new JProgressBar(0, 100);
        progressBar.setStringPainted(true);
        
        // Results table
        String[] columns = {"Algorithm", "Accuracy (%)", "Correctly Classified"};
        tableModel = new DefaultTableModel(columns, 0) {
            @Override
            public boolean isCellEditable(int row, int column) {
                return false;
            }
        };
        resultsTable = new JTable(tableModel);
        resultsTable.setRowHeight(25);
        
        // Best algorithm label
        bestAlgorithmLabel = new JLabel("Best Algorithm: -");
        bestAlgorithmLabel.setFont(new Font("Arial", Font.BOLD, 14));
        bestAlgorithmLabel.setForeground(new Color(0, 128, 0));
    }

    private void layoutComponents() {
        setLayout(new BorderLayout(10, 10));
        
        // Create classification panel
        JPanel classificationPanel = new JPanel(new BorderLayout(10, 10));

        // Top panel - Dataset selection
        JPanel topPanel = new JPanel(new BorderLayout(5, 5));
        topPanel.setBorder(BorderFactory.createTitledBorder("Dataset Selection"));
        
        JPanel datasetPanel = new JPanel(new BorderLayout(5, 5));
        datasetPanel.add(new JLabel("Dataset: "), BorderLayout.WEST);
        datasetPanel.add(datasetPathField, BorderLayout.CENTER);
        datasetPanel.add(browseButton, BorderLayout.EAST);
        
        topPanel.add(datasetPanel, BorderLayout.NORTH);
        topPanel.add(datasetInfoLabel, BorderLayout.CENTER);
        
        // Middle panel - Control
        JPanel middlePanel = new JPanel(new BorderLayout(5, 5));
        middlePanel.setBorder(BorderFactory.createTitledBorder("Classification Control"));
        middlePanel.add(startButton, BorderLayout.NORTH);
        middlePanel.add(progressBar, BorderLayout.CENTER);
        middlePanel.setPreferredSize(new Dimension(0, 100));

        // Bottom panel - Results
        JPanel bottomPanel = new JPanel(new BorderLayout(5, 5));
        bottomPanel.setBorder(BorderFactory.createTitledBorder("Results"));
        
        JScrollPane scrollPane = new JScrollPane(resultsTable);
        bottomPanel.add(scrollPane, BorderLayout.CENTER);
        bottomPanel.add(bestAlgorithmLabel, BorderLayout.SOUTH);
        
        // Combine top and middle panels
        JPanel topCombinedPanel = new JPanel(new BorderLayout(5, 5));
        topCombinedPanel.add(topPanel, BorderLayout.NORTH);
        topCombinedPanel.add(middlePanel, BorderLayout.CENTER);

        // Add all to classification panel
        classificationPanel.add(topCombinedPanel, BorderLayout.NORTH);
        classificationPanel.add(bottomPanel, BorderLayout.CENTER);

        // Create tabbed pane
        tabbedPane = new JTabbedPane();
        tabbedPane.addTab("Classification", classificationPanel);

        // Add tabbed pane to frame
        add(tabbedPane, BorderLayout.CENTER);
    }

    private void selectDataset() {
        JFileChooser fileChooser = new JFileChooser();
        fileChooser.setFileFilter(new javax.swing.filechooser.FileFilter() {
            @Override
            public boolean accept(File f) {
                return f.isDirectory() || f.getName().toLowerCase().endsWith(".arff");
            }
            
            @Override
            public String getDescription() {
                return "ARFF Files (*.arff)";
            }
        });
        
        int result = fileChooser.showOpenDialog(this);
        if (result == JFileChooser.APPROVE_OPTION) {
            selectedDataset = fileChooser.getSelectedFile();
            datasetPathField.setText(selectedDataset.getAbsolutePath());
            loadDatasetInfo();
            startButton.setEnabled(true);
        }
    }

    private void loadDatasetInfo() {
        try {
            DataProcessor processor = new DataProcessor(selectedDataset.getAbsolutePath());
            int instances = processor.getNumInstances();
            int attributes = processor.getNumAttributes();
            datasetInfoLabel.setText(String.format(
                "Instances: %d  |  Attributes: %d  |  Ready to classify", 
                instances, attributes
            ));
        } catch (Exception e) {
            datasetInfoLabel.setText("Error loading dataset: " + e.getMessage());
            startButton.setEnabled(false);
        }
    }

    private void startClassification() {
        startButton.setEnabled(false);
        browseButton.setEnabled(false);
        tableModel.setRowCount(0);
        bestAlgorithmLabel.setText("Best Algorithm: Processing...");
        progressBar.setValue(0);
        
        // Run classification in background thread
        SwingWorker<List<Result>, Integer> worker = new SwingWorker<>() {
            @Override
            protected List<Result> doInBackground() throws Exception {
                classificationEngine = new ClassificationEngine(
                    selectedDataset.getAbsolutePath()
                );
                
                classificationEngine.setProgressListener(progress -> {
                    publish(progress);
                });
                
                return classificationEngine.runAllClassifications();
            }
            
            @Override
            protected void process(List<Integer> chunks) {
                if (!chunks.isEmpty()) {
                    progressBar.setValue(chunks.get(chunks.size() - 1));
                }
            }
            
            @Override
            protected void done() {
                try {
                    List<Result> results = get();
                    displayResults(results);

                    // Add Discover panel after successful classification
                    if (classificationEngine != null && classificationEngine.isTrained()) {
                        // Remove existing Discover tab if present
                        for (int i = 0; i < tabbedPane.getTabCount(); i++) {
                            if (tabbedPane.getTitleAt(i).equals("Discover")) {
                                tabbedPane.removeTabAt(i);
                                break;
                            }
                        }

                        // Add new Discover panel
                        DiscoverPanel discoverPanel = new DiscoverPanel(classificationEngine);
                        tabbedPane.addTab("Discover", discoverPanel);

                        // Switch to Discover tab
                        tabbedPane.setSelectedIndex(1);
                    }
                } catch (Exception e) {
                    JOptionPane.showMessageDialog(
                        Main.this,
                        "Error during classification: " + e.getMessage(),
                        "Error",
                        JOptionPane.ERROR_MESSAGE
                    );
                    e.printStackTrace();
                } finally {
                    startButton.setEnabled(true);
                    browseButton.setEnabled(true);
                    progressBar.setValue(100);
                }
            }
        };
        
        worker.execute();
    }

    private void displayResults(List<Result> results) {
        // Sort by accuracy descending
        results.sort((r1, r2) -> Double.compare(r2.getAccuracy(), r1.getAccuracy()));
        
        // Add to table
        for (Result result : results) {
            tableModel.addRow(new Object[]{
                result.getAlgorithmName(),
                String.format("%.2f", result.getAccuracy()),
                result.getCorrectlyClassified() + " / " + result.getTotalInstances()
            });
        }
        
        // Highlight best
        if (!results.isEmpty()) {
            Result best = results.get(0);
            bestAlgorithmLabel.setText(String.format(
                "Best Algorithm: %s (%.2f%% accuracy)",
                best.getAlgorithmName(),
                best.getAccuracy()
            ));
            resultsTable.setRowSelectionInterval(0, 0);
        }
    }

    public static void main(String[] args) {
        try {
            UIManager.setLookAndFeel(UIManager.getSystemLookAndFeelClassName());
        } catch (Exception e) {
            e.printStackTrace();
        }
        
        SwingUtilities.invokeLater(() -> {
            Main frame = new Main();
            frame.setVisible(true);
        });
    }
}
