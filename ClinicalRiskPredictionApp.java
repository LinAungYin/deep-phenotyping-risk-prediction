import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * ClinicalRiskPredictionApp.java
 *
 * Project: Computational Deep Phenotyping for Clinical Risk Prediction
 * Description:
 * This Java application simulates the end-to-end process of using 'deep phenotypes'
 * (complex, derived clinical features) to train a Logistic Regression model for
 * binary risk prediction (e.g., High vs. Low risk for a specific event).
 * The code is entirely self-contained, demonstrating strong Java fundamentals,
 * data manipulation (normalization), and the implementation of a core ML algorithm
 * (Logistic Regression with Gradient Descent) from first principles.
 *
 * Target Audience: PhD Application Technical Showcase (NUS Yong Loo Lin)
 *
 * Phenotypes used: Age, Glucose Level, Comorbidity Score, Systolic BP.
 * Outcome: Risk (0 = Low, 1 = High).
 */
public class ClinicalRiskPredictionApp {

    public static void main(String[] args) {
        System.out.println("--- Computational Deep Phenotyping for Clinical Risk Prediction Showcase ---");

        // 1. Simulate Deep Phenotype Data Generation
        DeepPhenotypingEngine engine = new DeepPhenotypingEngine();
        List<PatientRecord> rawData = engine.generateSyntheticData(500);
        System.out.printf("Dataset Size: %d records generated.\n", rawData.size());
        
        // 2. Prepare and Normalize Data for Model Training
        // The predictor features (X) are extracted and the target labels (Y) are separated.
        double[][] X = new double[rawData.size()][4]; // 4 features: Age, Glucose, Comorbidity, BP
        double[] Y = new double[rawData.size()];
        
        // Find max values for simple min-max normalization
        double maxAge = 0, maxGlucose = 0, maxComorb = 0, maxBP = 0;
        for (PatientRecord p : rawData) {
            maxAge = Math.max(maxAge, p.getAge());
            maxGlucose = Math.max(maxGlucose, p.getGlucoseLevel());
            maxComorb = Math.max(maxComorb, p.getComorbidityScore());
            maxBP = Math.max(maxBP, p.getSystolicBP());
        }

        for (int i = 0; i < rawData.size(); i++) {
            PatientRecord p = rawData.get(i);
            // Simple normalization (scaling by max) to improve Gradient Descent convergence
            X[i][0] = p.getAge() / maxAge;
            X[i][1] = p.getGlucoseLevel() / maxGlucose;
            X[i][2] = p.getComorbidityScore() / maxComorb;
            X[i][3] = p.getSystolicBP() / maxBP;
            Y[i] = p.getRiskOutcome();
        }

        // 3. Initialize and Train the Risk Prediction Model
        int numFeatures = X[0].length; // The number of features (4)
        LogisticRegressionModel model = new LogisticRegressionModel(numFeatures);

        System.out.println("\n--- Starting Model Training (Logistic Regression via Gradient Descent) ---");
        int iterations = 10000;
        double learningRate = 0.1;
        
        model.train(X, Y, learningRate, iterations);

        // 4. Evaluation and Prediction Demonstration
        System.out.println("\n--- Model Evaluation and Demonstration ---");
        
        // Test on the training data (for demonstration purposes)
        int correctPredictions = 0;
        for (int i = 0; i < rawData.size(); i++) {
            double predictedProb = model.predict(X[i]);
            int predictedClass = predictedProb >= 0.5 ? 1 : 0;
            
            if (predictedClass == Y[i]) {
                correctPredictions++;
            }
        }
        
        double accuracy = (double) correctPredictions / rawData.size() * 100;
        System.out.printf("Training Accuracy: %.2f%% (%d/%d)\n", 
                          accuracy, correctPredictions, rawData.size());
        
        // 5. Demonstrate Prediction on a New (Unseen) Patient
        // Define a new patient's raw features
        double newPatientAge = 72;
        double newPatientGlucose = 145.0; // High
        double newPatientComorb = 0.85;  // High
        double newPatientBP = 150.0;
        
        // Normalize the new patient's features using the MAX values from the training data
        double[] newPatientFeaturesNormalized = new double[]{
            newPatientAge / maxAge,
            newPatientGlucose / maxGlucose,
            newPatientComorb / maxComorb,
            newPatientBP / maxBP
        };
        
        double riskProbability = model.predict(newPatientFeaturesNormalized);
        String riskLevel = riskProbability >= 0.5 ? "HIGH" : "LOW";
        
        System.out.println("\n--- New Patient Risk Prediction ---");
        System.out.printf("Patient Phenotypes: Age=%.0f, Glucose=%.1f, Comorbidity=%.2f, BP=%.1f\n",
                          newPatientAge, newPatientGlucose, newPatientComorb, newPatientBP);
        System.out.printf("Predicted Risk Probability (P(Risk=1)): %.4f\n", riskProbability);
        System.out.printf("Clinical Risk Prediction: %s Risk\n", riskLevel);
    }
}

/**
 * Represents a single patient's derived 'Deep Phenotype' record.
 * This class abstracts the complex process of data extraction and feature engineering.
 */
class PatientRecord {
    private final int age;
    private final double glucoseLevel; // Proxy for metabolic health
    private final double comorbidityScore; // Derived score (e.g., Charlson Index, adjusted for severity)
    private final double systolicBP;
    private final int riskOutcome; // Target: 0 (Low Risk) or 1 (High Risk)

    public PatientRecord(int age, double glucoseLevel, double comorbidityScore, double systolicBP, int riskOutcome) {
        this.age = age;
        this.glucoseLevel = glucoseLevel;
        this.comorbidityScore = comorbidityScore;
        this.systolicBP = systolicBP;
        this.riskOutcome = riskOutcome;
    }

    public int getAge() { return age; }
    public double getGlucoseLevel() { return glucoseLevel; }
    public double getComorbidityScore() { return comorbidityScore; }
    public double getSystolicBP() { return systolicBP; }
    public int getRiskOutcome() { return riskOutcome; }
}

/**
 * Simulates the Deep Phenotyping process, transforming raw data into structured features.
 * In a real-world scenario, this would involve NLP on clinical notes, time-series analysis, etc.
 */
class DeepPhenotypingEngine {
    private final Random rand = new Random();

    /**
     * Generates synthetic patient records where the risk outcome is correlated
     * with higher values in the features (Age, Glucose, Comorbidity, BP).
     */
    public List<PatientRecord> generateSyntheticData(int count) {
        List<PatientRecord> data = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            // Base values
            int age = rand.nextInt(50) + 30; // 30-79
            double glucoseLevel = rand.nextDouble() * 50 + 90; // 90.0 - 139.9
            double comorbidityScore = rand.nextDouble() * 0.9 + 0.1; // 0.1 - 1.0
            double systolicBP = rand.nextDouble() * 40 + 110; // 110.0 - 149.9

            int risk = 0;
            // Introduce positive correlation for high risk (1)
            if (age > 65 && rand.nextDouble() < 0.6) risk = 1;
            if (glucoseLevel > 120 && rand.nextDouble() < 0.7) risk = 1;
            if (comorbidityScore > 0.7 && rand.nextDouble() < 0.8) risk = 1;
            if (systolicBP > 140 && rand.nextDouble() < 0.6) risk = 1;
            
            // Random chance for low risk patients to be high risk (noise)
            if (rand.nextDouble() < 0.15) risk = 1 - risk;

            // Simple rule: If enough factors are high, enforce high risk
            int highFactorCount = 0;
            if (age > 60) highFactorCount++;
            if (glucoseLevel > 130) highFactorCount++;
            if (comorbidityScore > 0.8) highFactorCount++;
            
            if (highFactorCount >= 2) risk = 1;


            data.add(new PatientRecord(age, glucoseLevel, comorbidityScore, systolicBP, risk));
        }
        return data;
    }
}

/**
 * Implements a Logistic Regression Classifier using Batch Gradient Descent.
 * This class handles training and prediction for binary classification (0 or 1).
 */
class LogisticRegressionModel {
    // We add +1 for the bias (intercept) term
    private final int numFeatures; 
    private double[] weights; // Stores the coefficients (theta) and the bias (weights[0])

    public LogisticRegressionModel(int numInputFeatures) {
        // The total number of weights is (numInputFeatures + 1) for the bias term
        this.numFeatures = numInputFeatures + 1; 
        this.weights = new double[this.numFeatures];
        // Initialize weights to small random values (or zero for simplicity)
        for (int i = 0; i < this.numFeatures; i++) {
            this.weights[i] = 0.0; // Simple initialization
        }
    }

    /**
     * The sigmoid (or logistic) function: g(z) = 1 / (1 + e^(-z))
     * Maps any real number to a probability between 0 and 1.
     */
    private double sigmoid(double z) {
        return 1.0 / (1.0 + Math.exp(-z));
    }

    /**
     * Predicts the probability of the outcome being 1 for a given feature vector.
     * h(x) = sigmoid(w0*1 + w1*x1 + w2*x2 + ...)
     */
    public double predict(double[] features) {
        // Start with the bias term (weights[0] * 1)
        double z = this.weights[0]; 
        
        // Add the contribution of all features: w_i * x_i
        for (int i = 0; i < features.length; i++) {
            z += this.weights[i + 1] * features[i]; // features[i] corresponds to weights[i+1]
        }
        return sigmoid(z);
    }

    /**
     * Trains the model weights using Batch Gradient Descent.
     * @param X The feature matrix (normalized input data)
     * @param Y The target vector (0 or 1)
     * @param learningRate The step size for weight updates (alpha)
     * @param iterations The number of training epochs
     */
    public void train(double[][] X, double[] Y, double learningRate, int iterations) {
        int m = X.length; // Number of training examples
        
        for (int iter = 0; iter < iterations; iter++) {
            // Calculate the gradient (sum of errors * features)
            double[] gradient = new double[this.numFeatures];

            for (int i = 0; i < m; i++) { // Loop over each training example
                double predictedProb = predict(X[i]);
                double error = predictedProb - Y[i]; // Difference between prediction and actual

                // Update gradient for the Bias term (j=0)
                gradient[0] += error; 

                // Update gradient for the feature terms (j=1 to numFeatures-1)
                for (int j = 1; j < this.numFeatures; j++) {
                    gradient[j] += error * X[i][j - 1]; // X[i][j-1] is the feature corresponding to weight w_j
                }
            }

            // Update weights using the calculated gradient
            for (int j = 0; j < this.numFeatures; j++) {
                // Weight update formula: w_j = w_j - alpha * (1/m) * gradient_j
                this.weights[j] -= learningRate * (1.0 / m) * gradient[j];
            }

            // Optional: Print loss every N iterations to monitor convergence
            if (iter % (iterations / 10) == 0) {
                double cost = calculateCost(X, Y);
                System.out.printf("Iteration %d: Cost (Log Loss) = %.6f\n", iter, cost);
            }
        }
        
        System.out.println("Training Complete.");
        System.out.printf("Final Weights (w0, w1...): [Bias: %.4f,", this.weights[0]);
        for (int i = 1; i < this.numFeatures; i++) {
            System.out.printf(" w%d: %.4f%s", i, this.weights[i], (i == this.numFeatures - 1 ? "" : ","));
        }
        System.out.println("]");
    }

    /**
     * Calculates the Binary Cross-Entropy (Log Loss) cost function.
     * J(w) = -1/m * Sum[ y * log(h(x)) + (1-y) * log(1-h(x)) ]
     */
    private double calculateCost(double[][] X, double[] Y) {
        int m = X.length;
        double cost = 0.0;
        
        for (int i = 0; i < m; i++) {
            double h_x = predict(X[i]);
            // Ensure no log(0) calculation
            h_x = Math.max(1e-15, Math.min(1.0 - 1e-15, h_x)); 
            
            cost += Y[i] * Math.log(h_x) + (1.0 - Y[i]) * Math.log(1.0 - h_x);
        }
        
        return -cost / m;
    }
}
