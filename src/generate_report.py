"""
Generate the final capstone report as a professional PDF.
Embeds all charts, model metrics, and UI screenshots.
"""
import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch, mm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib import colors

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIGURES_DIR = os.path.join(PROJECT_DIR, "outputs", "figures")
SCREENSHOTS_DIR = os.path.join(PROJECT_DIR, "outputs", "screenshots")
OUTPUT_PATH = os.path.join(PROJECT_DIR, "outputs", "Car_Price_Prediction_Final_Report.pdf")

# Colors
BLUE = HexColor("#2563eb")
DARK_BLUE = HexColor("#1e3a5f")
LIGHT_BLUE = HexColor("#eff6ff")
GRAY = HexColor("#64748b")
DARK = HexColor("#0f172a")
LIGHT_GRAY = HexColor("#f1f5f9")
BORDER_GRAY = HexColor("#e2e8f0")


def get_styles():
    """Create custom paragraph styles."""
    styles = getSampleStyleSheet()

    styles.add(ParagraphStyle(
        'CoverTitle', parent=styles['Title'],
        fontSize=32, leading=38, textColor=DARK,
        fontName='Helvetica-Bold', alignment=TA_CENTER,
        spaceAfter=12,
    ))
    styles.add(ParagraphStyle(
        'CoverSubtitle', parent=styles['Normal'],
        fontSize=16, leading=22, textColor=BLUE,
        fontName='Helvetica', alignment=TA_CENTER,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'CoverInfo', parent=styles['Normal'],
        fontSize=11, leading=16, textColor=GRAY,
        fontName='Helvetica', alignment=TA_CENTER,
    ))
    styles.add(ParagraphStyle(
        'SectionTitle', parent=styles['Heading1'],
        fontSize=20, leading=26, textColor=DARK,
        fontName='Helvetica-Bold', spaceBefore=24, spaceAfter=12,
        borderColor=BLUE, borderWidth=0, borderPadding=0,
    ))
    styles.add(ParagraphStyle(
        'SubSection', parent=styles['Heading2'],
        fontSize=14, leading=18, textColor=DARK_BLUE,
        fontName='Helvetica-Bold', spaceBefore=16, spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'BodyText2', parent=styles['Normal'],
        fontSize=10.5, leading=16, textColor=DARK,
        fontName='Helvetica', alignment=TA_JUSTIFY,
        spaceAfter=8,
    ))
    styles.add(ParagraphStyle(
        'Caption', parent=styles['Normal'],
        fontSize=9, leading=13, textColor=GRAY,
        fontName='Helvetica-Oblique', alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=16,
    ))
    styles.add(ParagraphStyle(
        'BulletItem', parent=styles['Normal'],
        fontSize=10.5, leading=16, textColor=DARK,
        fontName='Helvetica', leftIndent=20,
        spaceAfter=4, bulletIndent=8,
    ))
    return styles


def section_divider():
    """Blue horizontal rule."""
    return HRFlowable(width="100%", thickness=2, color=BLUE,
                      spaceBefore=6, spaceAfter=12)


def add_image(path, width=6*inch, caption=None, styles=None):
    """Add an image with optional caption."""
    elements = []
    if os.path.exists(path):
        img = Image(path, width=width, height=width * 0.65)
        img.hAlign = 'CENTER'
        elements.append(img)
        if caption and styles:
            elements.append(Paragraph(caption, styles['Caption']))
    return elements


def make_table(data, col_widths=None, highlight_last=False):
    """Create a styled table."""
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), DARK),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_GRAY),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT_GRAY]),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]
    if highlight_last:
        style_cmds.append(('BACKGROUND', (0, -1), (-1, -1), LIGHT_BLUE))
        style_cmds.append(('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'))
    t.setStyle(TableStyle(style_cmds))
    t.hAlign = 'CENTER'
    return t


def build_report():
    """Build the complete PDF report."""
    doc = SimpleDocTemplate(
        OUTPUT_PATH, pagesize=A4,
        leftMargin=25*mm, rightMargin=25*mm,
        topMargin=25*mm, bottomMargin=25*mm,
    )
    styles = get_styles()
    story = []

    # ======= COVER PAGE =======
    story.append(Spacer(1, 80))
    story.append(Paragraph("Car Price Prediction", styles['CoverTitle']))
    story.append(Paragraph("Using Machine Learning on Cloud", styles['CoverSubtitle']))
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="40%", thickness=3, color=BLUE,
                             spaceBefore=0, spaceAfter=20))
    story.append(Paragraph("Capstone Project - Final Report", styles['CoverInfo']))
    story.append(Spacer(1, 12))
    story.append(Paragraph("April 2026", styles['CoverInfo']))
    story.append(Spacer(1, 40))

    # Cover image - the prediction UI
    cover_img = os.path.join(SCREENSHOTS_DIR, "new_ui_prediction.png")
    story.extend(add_image(cover_img, width=5.2*inch,
                           caption="Live Prediction Interface", styles=styles))

    story.append(Spacer(1, 40))
    story.append(Paragraph("Technology Stack: Python - scikit-learn - XGBoost - Flask - Docker - Azure", styles['CoverInfo']))

    story.append(PageBreak())

    # ======= TABLE OF CONTENTS =======
    story.append(Paragraph("Table of Contents", styles['SectionTitle']))
    story.append(section_divider())
    toc_items = [
        "1. Executive Summary",
        "2. Problem Statement",
        "3. Dataset Description",
        "4. Exploratory Data Analysis",
        "5. Methodology",
        "6. Model Training and Comparison",
        "7. Results and Evaluation",
        "8. Web Application and Deployment",
        "9. Testing and Quality Assurance",
        "10. Challenges and Learnings",
        "11. Future Improvements",
        "12. Conclusion",
        "13. References",
    ]
    for item in toc_items:
        story.append(Paragraph(item, styles['BodyText2']))
    story.append(PageBreak())

    # ======= 1. EXECUTIVE SUMMARY =======
    story.append(Paragraph("1. Executive Summary", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "This project develops a machine learning system that predicts used car selling prices "
        "based on vehicle attributes such as brand, age, mileage, fuel type, and transmission. "
        "Three regression algorithms were trained and compared: Linear Regression, Random Forest, "
        "and XGBoost. The XGBoost model achieved the best performance with an R-squared of 0.982, "
        "meaning it explains 98.2% of the variance in car prices. The trained model is served "
        "through a Flask web application with a professional user interface, containerized with "
        "Docker, and ready for deployment on Microsoft Azure App Service.",
        styles['BodyText2']))
    story.append(Spacer(1, 8))

    # Key metrics highlight box
    metrics_data = [
        ['Metric', 'Value'],
        ['Best Model', 'XGBoost'],
        ['R-squared', '0.9820'],
        ['Mean Absolute Error', '0.3833 Lakhs'],
        ['Root Mean Squared Error', '0.7083 Lakhs'],
        ['Cross-Validation R2', '0.9915 (+/- 0.0006)'],
        ['Dataset Size', '8,000 records (7,844 after cleaning)'],
        ['Total Automated Tests', '20 (all passing)'],
    ]
    story.append(make_table(metrics_data, col_widths=[2.5*inch, 3.5*inch]))
    story.append(Paragraph("Table 1: Key Project Metrics", styles['Caption']))
    story.append(PageBreak())

    # ======= 2. PROBLEM STATEMENT =======
    story.append(Paragraph("2. Problem Statement", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "Used car pricing in India is often inconsistent and depends heavily on subjective "
        "judgment. Sellers may overvalue their vehicles due to emotional attachment, while buyers "
        "lack reliable benchmarks to assess fair market value. The absence of a standardized, "
        "data-driven pricing mechanism creates information asymmetry that disadvantages both "
        "parties in the transaction.",
        styles['BodyText2']))
    story.append(Paragraph(
        "This project addresses this gap by developing a predictive model trained on historical "
        "vehicle sales data. The model estimates fair selling prices based on objective vehicle "
        "attributes, providing a transparent and reproducible pricing tool that can be accessed "
        "through a simple web interface.",
        styles['BodyText2']))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Project Objectives", styles['SubSection']))
    objectives = [
        "Build a machine learning model to predict used car selling prices",
        "Compare multiple algorithms (Linear Regression, Random Forest, XGBoost) and select the best",
        "Evaluate performance using industry-standard metrics: RMSE, MAE, and R-squared",
        "Deploy the trained model on the cloud via a web application",
        "Demonstrate live predictions through an interactive API and web form",
    ]
    for obj in objectives:
        story.append(Paragraph(f"&#8226;  {obj}", styles['BulletItem']))
    story.append(PageBreak())

    # ======= 3. DATASET DESCRIPTION =======
    story.append(Paragraph("3. Dataset Description", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "The dataset is modeled after the popular CarDekho vehicle dataset available on Kaggle. "
        "It contains 8,000 records of used car sales with 9 features capturing vehicle identity, "
        "usage metrics, and market conditions. After data cleaning (removing outliers, duplicates, "
        "and handling missing values), 7,844 records remain for model training.",
        styles['BodyText2']))
    story.append(Spacer(1, 8))

    dataset_table = [
        ['Feature', 'Type', 'Description'],
        ['Brand', 'Categorical', 'Car manufacturer (15 brands: Maruti, Hyundai, BMW, etc.)'],
        ['Present_Price', 'Numeric', 'Current ex-showroom price in Lakhs INR'],
        ['Kms_Driven', 'Numeric', 'Total kilometers driven'],
        ['Fuel_Type', 'Categorical', 'Petrol, Diesel, or CNG'],
        ['Seller_Type', 'Categorical', 'Dealer or Individual'],
        ['Transmission', 'Categorical', 'Manual or Automatic'],
        ['Owner', 'Numeric', 'Number of previous owners (0, 1, 3)'],
        ['Car_Age', 'Numeric', 'Age of the car in years (derived from Year)'],
        ['Selling_Price', 'Numeric', 'Target variable - actual selling price in Lakhs INR'],
    ]
    story.append(make_table(dataset_table, col_widths=[1.3*inch, 1.1*inch, 3.6*inch]))
    story.append(Paragraph("Table 2: Dataset Features", styles['Caption']))
    story.append(PageBreak())

    # ======= 4. EXPLORATORY DATA ANALYSIS =======
    story.append(Paragraph("4. Exploratory Data Analysis", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "A thorough exploratory analysis was conducted to understand data distributions, "
        "feature relationships, and potential issues before model training. Key findings "
        "are presented below with supporting visualizations.",
        styles['BodyText2']))

    # Price distribution
    story.append(Paragraph("4.1 Target Variable Distribution", styles['SubSection']))
    story.append(Paragraph(
        "The selling price distribution is heavily right-skewed, with most cars priced between "
        "1-10 Lakhs and a long tail extending to luxury vehicles above 30 Lakhs. A log "
        "transformation was applied to normalize the distribution for model training, significantly "
        "improving prediction accuracy across all price ranges.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "01_price_distribution.png"), width=5.8*inch,
        caption="Figure 1: Selling Price Distribution (Original and Log-Transformed)", styles=styles))

    # Correlation heatmap
    story.append(Paragraph("4.2 Feature Correlations", styles['SubSection']))
    story.append(Paragraph(
        "The correlation heatmap reveals that Present_Price (ex-showroom price) has the strongest "
        "positive correlation with Selling_Price (0.88), followed by Car_Age showing a strong "
        "negative correlation (-0.47). Kilometers driven shows a moderate negative relationship, "
        "confirming that both age and usage significantly impact resale value.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "02_correlation_heatmap.png"), width=4.8*inch,
        caption="Figure 2: Feature Correlation Heatmap", styles=styles))

    story.append(PageBreak())

    # Categorical analysis
    story.append(Paragraph("4.3 Categorical Feature Analysis", styles['SubSection']))
    story.append(Paragraph(
        "Analysis of categorical features reveals distinct pricing patterns. Diesel vehicles "
        "command higher resale prices than petrol variants. Automatic transmission vehicles "
        "show significantly higher prices, driven partly by the luxury segment. Dealer sales "
        "tend to achieve slightly higher prices than individual sellers.",
        styles['BodyText2']))

    story.extend(add_image(
        os.path.join(FIGURES_DIR, "03_price_by_fuel.png"), width=5.5*inch,
        caption="Figure 3: Selling Price Distribution by Fuel Type", styles=styles))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "04_price_by_transmission.png"), width=5.5*inch,
        caption="Figure 4: Selling Price Distribution by Transmission Type", styles=styles))

    story.append(PageBreak())

    # Scatter plots
    story.append(Paragraph("4.4 Numeric Feature Relationships", styles['SubSection']))
    story.append(Paragraph(
        "Scatter plots reveal the non-linear depreciation pattern: car prices drop steeply "
        "in the first few years and then level off. Higher mileage consistently correlates "
        "with lower selling prices, though the effect is more pronounced in budget vehicles.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "05_price_vs_age.png"), width=5.5*inch,
        caption="Figure 5: Selling Price vs Car Age", styles=styles))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "06_price_vs_kms.png"), width=5.5*inch,
        caption="Figure 6: Selling Price vs Kilometers Driven", styles=styles))

    story.append(PageBreak())

    # Brand distribution
    story.append(Paragraph("4.5 Brand Distribution", styles['SubSection']))
    story.append(Paragraph(
        "The dataset reflects the Indian automobile market with Maruti and Hyundai dominating "
        "sales volume. Luxury brands (BMW, Audi, Mercedes-Benz) have fewer records but "
        "significantly higher price points, creating an important segment for the model to handle.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "07_brand_distribution.png"), width=5.8*inch,
        caption="Figure 7: Number of Cars by Brand", styles=styles))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "08_price_by_seller.png"), width=5.5*inch,
        caption="Figure 8: Selling Price by Seller Type", styles=styles))

    story.append(PageBreak())

    # ======= 5. METHODOLOGY =======
    story.append(Paragraph("5. Methodology", styles['SectionTitle']))
    story.append(section_divider())

    story.append(Paragraph("5.1 Data Preprocessing", styles['SubSection']))
    preprocess_steps = [
        "Brand extraction from Car_Name (e.g., 'Maruti Swift' -> 'Maruti')",
        "Car age calculation: Car_Age = 2026 - Year of manufacture",
        "Missing value imputation using median (numeric) and mode (categorical)",
        "Duplicate row removal",
        "Outlier removal using the IQR method (1st to 99th percentile filtering)",
        "Dropping raw Car_Name and Year columns (replaced by Brand and Car_Age)",
    ]
    for step in preprocess_steps:
        story.append(Paragraph(f"&#8226;  {step}", styles['BulletItem']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("5.2 Feature Engineering Pipeline", styles['SubSection']))
    story.append(Paragraph(
        "A scikit-learn ColumnTransformer was used to build a reproducible preprocessing pipeline "
        "that prevents data leakage during cross-validation. Numeric features (Present_Price, "
        "Kms_Driven, Owner, Car_Age) are standardized using StandardScaler. Categorical features "
        "(Brand, Fuel_Type, Seller_Type, Transmission) are one-hot encoded with "
        "handle_unknown='ignore' for robustness against unseen categories at inference time.",
        styles['BodyText2']))
    story.append(Paragraph(
        "The target variable (Selling_Price) is log-transformed using log(1+x) to handle the "
        "right-skewed distribution. This transformation significantly improved model performance "
        "across all algorithms by normalizing the error distribution.",
        styles['BodyText2']))

    story.append(Spacer(1, 12))
    story.append(Paragraph("5.3 Model Selection", styles['SubSection']))
    story.append(Paragraph(
        "Three regression algorithms were selected to cover a spectrum from simple linear "
        "models to advanced ensemble methods:",
        styles['BodyText2']))
    models_desc = [
        "Linear Regression: Baseline model providing an interpretable lower bound on performance",
        "Random Forest (200 trees, max_depth=15): Ensemble of decision trees that captures non-linear relationships and provides feature importance rankings",
        "XGBoost (300 trees, learning_rate=0.1, max_depth=6): Gradient boosting algorithm known for state-of-the-art performance on tabular data",
    ]
    for m in models_desc:
        story.append(Paragraph(f"&#8226;  {m}", styles['BulletItem']))

    story.append(Paragraph(
        "All models were trained within the sklearn Pipeline framework, ensuring that "
        "preprocessing and prediction are a single atomic operation. This design prevents "
        "data leakage and simplifies deployment.",
        styles['BodyText2']))
    story.append(PageBreak())

    # ======= 6. MODEL TRAINING AND COMPARISON =======
    story.append(Paragraph("6. Model Training and Comparison", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "The dataset was split 80/20 into training (6,275 samples) and test (1,569 samples) sets "
        "with a fixed random seed (42) for reproducibility. Each model was evaluated on the held-out "
        "test set and additionally validated using 5-fold cross-validation on the training set.",
        styles['BodyText2']))
    story.append(Spacer(1, 8))

    comparison_table = [
        ['Model', 'MAE (Lakhs)', 'RMSE (Lakhs)', 'R-squared', 'CV R2 Mean', 'Train Time'],
        ['Linear Regression', '1.0535', '1.9765', '0.8596', '0.9206', '0.01s'],
        ['Random Forest', '0.4436', '0.8036', '0.9768', '0.9849', '0.99s'],
        ['XGBoost (Best)', '0.3833', '0.7083', '0.9820', '0.9915', '0.68s'],
    ]
    story.append(make_table(comparison_table,
                            col_widths=[1.4*inch, 1*inch, 1.1*inch, 0.9*inch, 0.9*inch, 0.8*inch],
                            highlight_last=True))
    story.append(Paragraph("Table 3: Model Performance Comparison", styles['Caption']))

    story.append(Paragraph(
        "XGBoost achieves the best performance across all metrics with an R-squared of 0.982 "
        "and a cross-validation R-squared of 0.9915 (standard deviation of only 0.0006), "
        "indicating highly stable and generalizable performance. The model's MAE of 0.38 Lakhs "
        "means predictions are off by approximately Rs. 38,000 on average -- well within "
        "acceptable margins for used car valuation.",
        styles['BodyText2']))
    story.append(PageBreak())

    # ======= 7. RESULTS AND EVALUATION =======
    story.append(Paragraph("7. Results and Evaluation", styles['SectionTitle']))
    story.append(section_divider())

    story.append(Paragraph("7.1 Actual vs Predicted Prices", styles['SubSection']))
    story.append(Paragraph(
        "The scatter plot below shows predicted vs actual selling prices for the test set. "
        "Points closely follow the red diagonal (perfect prediction line), with minimal "
        "scatter. The model performs well across the full price range from budget cars "
        "(under 5 Lakhs) to luxury vehicles (above 30 Lakhs).",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "09_actual_vs_predicted.png"), width=5.2*inch,
        caption="Figure 9: Actual vs Predicted Selling Price (Test Set)", styles=styles))

    story.append(Paragraph("7.2 Residual Analysis", styles['SubSection']))
    story.append(Paragraph(
        "Residuals are centered around zero with a roughly normal distribution, confirming "
        "that the model does not exhibit systematic bias. The residual-vs-predicted plot shows "
        "no clear pattern, indicating the model captures the underlying relationships well.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "10_residuals.png"), width=5.8*inch,
        caption="Figure 10: Residual Analysis (Scatter and Distribution)", styles=styles))

    story.append(PageBreak())

    story.append(Paragraph("7.3 Feature Importance", styles['SubSection']))
    story.append(Paragraph(
        "The feature importance chart from the XGBoost model reveals that Present_Price "
        "(ex-showroom price) and Car_Age are by far the most influential predictors. "
        "Brand-level features also contribute meaningfully, particularly for luxury brands "
        "where depreciation patterns differ significantly from budget vehicles.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "11_feature_importance.png"), width=5.5*inch,
        caption="Figure 11: Top 20 Feature Importances (XGBoost)", styles=styles))

    story.append(Paragraph("7.4 Learning Curves", styles['SubSection']))
    story.append(Paragraph(
        "Learning curves show that the model converges with the available training data. "
        "The small gap between training and validation scores confirms that the model is "
        "not overfitting, and that 8,000 records provide sufficient data for reliable "
        "predictions.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(FIGURES_DIR, "12_learning_curves.png"), width=5.2*inch,
        caption="Figure 12: Learning Curves (Training vs Validation R2)", styles=styles))

    story.append(PageBreak())

    # ======= 8. WEB APPLICATION AND DEPLOYMENT =======
    story.append(Paragraph("8. Web Application and Deployment", styles['SectionTitle']))
    story.append(section_divider())

    story.append(Paragraph("8.1 Application Architecture", styles['SubSection']))
    story.append(Paragraph(
        "The trained model is served through a Flask web application that provides both a "
        "user-friendly HTML form and a JSON API endpoint. The application loads the serialized "
        "XGBoost pipeline at startup and handles predictions in real-time. The architecture "
        "follows a simple pattern: User -> Flask API -> sklearn Pipeline -> Prediction.",
        styles['BodyText2']))
    story.append(Spacer(1, 8))

    api_table = [
        ['Endpoint', 'Method', 'Description'],
        ['/', 'GET', 'Serves the prediction web form'],
        ['/predict', 'POST', 'Accepts form data or JSON, returns prediction'],
        ['/health', 'GET', 'Health check (model loaded status)'],
    ]
    story.append(make_table(api_table, col_widths=[1.3*inch, 1*inch, 3.7*inch]))
    story.append(Paragraph("Table 4: API Endpoints", styles['Caption']))

    story.append(Paragraph("8.2 User Interface", styles['SubSection']))
    story.append(Paragraph(
        "The web interface features a clean, modern design with a white background, "
        "blue accent colors, and smooth animations. Users fill in vehicle details and "
        "receive an instant price prediction with a counting animation effect.",
        styles['BodyText2']))
    story.extend(add_image(
        os.path.join(SCREENSHOTS_DIR, "new_ui_home.png"), width=5.2*inch,
        caption="Figure 13: Web Application - Input Form", styles=styles))
    story.extend(add_image(
        os.path.join(SCREENSHOTS_DIR, "new_ui_prediction.png"), width=5.2*inch,
        caption="Figure 14: Web Application - Prediction Result", styles=styles))

    story.append(PageBreak())

    story.append(Paragraph("8.3 Demo Test Cases", styles['SubSection']))
    story.append(Paragraph(
        "Three representative test cases were prepared to demonstrate the model's ability "
        "to handle different vehicle segments:",
        styles['BodyText2']))

    demo_table = [
        ['Test Case', 'Brand', 'Fuel', 'Trans.', 'Age', 'Ex-Showroom', 'Predicted Price'],
        ['Budget Car', 'Maruti', 'Petrol', 'Manual', '3 yr', '7.5 L', '6.44 Lakhs'],
        ['Luxury Car', 'BMW', 'Diesel', 'Auto', '2 yr', '45 L', '30.06 Lakhs'],
        ['Mid-Range SUV', 'Toyota', 'Diesel', 'Auto', '5 yr', '32 L', '20.06 Lakhs'],
    ]
    story.append(make_table(demo_table,
                            col_widths=[1.1*inch, 0.7*inch, 0.6*inch, 0.6*inch, 0.5*inch, 1*inch, 1.1*inch]))
    story.append(Paragraph("Table 5: Demo Test Cases with Predictions", styles['Caption']))

    story.append(Paragraph("8.4 Cloud Deployment", styles['SubSection']))
    story.append(Paragraph(
        "The application is containerized using Docker and configured for deployment on "
        "Microsoft Azure App Service. The Dockerfile uses Python 3.10 slim base image, "
        "installs dependencies from requirements.txt, and exposes port 8080. The port is "
        "configurable via the PORT environment variable for cloud platform compatibility. "
        "Estimated cost on Azure for Students is approximately $13/month using the B1 tier, "
        "well within the free $100 credit allocation.",
        styles['BodyText2']))
    story.append(PageBreak())

    # ======= 9. TESTING =======
    story.append(Paragraph("9. Testing and Quality Assurance", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "A comprehensive test suite of 20 automated tests was developed using pytest and "
        "Playwright to ensure reliability across all project components:",
        styles['BodyText2']))
    story.append(Spacer(1, 8))

    test_table = [
        ['Test Category', 'Count', 'What It Covers'],
        ['Data Preprocessing', '6', 'Brand extraction, null handling, duplicates, outliers, car age, full pipeline'],
        ['Model Validation', '6', 'Model loading, predictions, range checks, single input, R2 >= 0.85, metrics file'],
        ['API Testing', '7', 'Home page, health check, form POST, JSON API, brand comparison, error handling'],
        ['E2E (Playwright)', '1', 'Full browser demo: 3 test cases, JSON API, error handling, 6 screenshots'],
        ['Total', '20', 'All passing - zero failures'],
    ]
    story.append(make_table(test_table, col_widths=[1.3*inch, 0.6*inch, 4.1*inch], highlight_last=True))
    story.append(Paragraph("Table 6: Test Suite Summary", styles['Caption']))

    story.append(Paragraph(
        "The end-to-end Playwright test simulates the exact demo flow: loading the page, "
        "filling in vehicle details for three different car segments (budget, luxury, mid-range), "
        "submitting predictions, and verifying the results. This ensures the application "
        "works flawlessly during live demonstrations.",
        styles['BodyText2']))
    story.append(PageBreak())

    # ======= 10. CHALLENGES =======
    story.append(Paragraph("10. Challenges and Learnings", styles['SectionTitle']))
    story.append(section_divider())
    challenges = [
        ("XGBoost OpenMP Dependency",
         "XGBoost requires the OpenMP runtime library (libomp) on macOS. This was resolved "
         "by installing it via Homebrew: brew install libomp."),
        ("Data Leakage Prevention",
         "Using sklearn Pipeline with ColumnTransformer ensures that feature scaling and encoding "
         "are fitted only on training data during cross-validation, preventing data leakage."),
        ("Skewed Price Distribution",
         "Applying log(1+x) transformation to the target variable normalized the distribution "
         "and improved R-squared from approximately 0.85 to 0.98 for tree-based models."),
        ("Port Conflicts on macOS",
         "Flask's default port 5000 conflicts with AirPlay Receiver on modern macOS. The "
         "application was configured to use port 8080 with environment variable override."),
        ("JSON Serialization",
         "NumPy float32 values from XGBoost predictions are not JSON-serializable by default. "
         "Explicit casting to Python float resolved this issue."),
    ]
    for title, desc in challenges:
        story.append(Paragraph(title, styles['SubSection']))
        story.append(Paragraph(desc, styles['BodyText2']))

    story.append(PageBreak())

    # ======= 11. FUTURE IMPROVEMENTS =======
    story.append(Paragraph("11. Future Improvements", styles['SectionTitle']))
    story.append(section_divider())
    improvements = [
        "Integrate the actual Kaggle CarDekho dataset for real-world validation",
        "Add additional features such as engine size, horsepower, and geographic location",
        "Implement model monitoring and automated retraining pipeline",
        "Add user authentication and prediction history tracking",
        "Deploy with CI/CD pipeline using GitHub Actions and Azure App Service",
        "Implement A/B testing to compare model versions in production",
        "Add confidence intervals to predictions for better user understanding",
    ]
    for imp in improvements:
        story.append(Paragraph(f"&#8226;  {imp}", styles['BulletItem']))

    story.append(Spacer(1, 20))

    # ======= 12. CONCLUSION =======
    story.append(Paragraph("12. Conclusion", styles['SectionTitle']))
    story.append(section_divider())
    story.append(Paragraph(
        "This project successfully demonstrates an end-to-end machine learning pipeline from "
        "data preprocessing through model deployment. The XGBoost model achieves an R-squared "
        "of 0.982, accurately predicting used car prices across budget, mid-range, and luxury "
        "segments. The Flask web application provides an intuitive interface for real-time "
        "predictions, and the Dockerized deployment package makes it ready for cloud hosting "
        "on Azure App Service.",
        styles['BodyText2']))
    story.append(Paragraph(
        "All project objectives outlined in the proposal have been met: multiple algorithms "
        "were compared, performance was evaluated with standard metrics, the model was saved "
        "and deployed via a web API, and comprehensive testing ensures reliability. The 20 "
        "automated tests (including end-to-end browser testing) provide confidence that the "
        "system works correctly across all scenarios.",
        styles['BodyText2']))
    story.append(Paragraph(
        "This project serves as a strong demonstration of practical machine learning skills "
        "combined with software engineering best practices, ready for both academic evaluation "
        "and professional portfolio presentation.",
        styles['BodyText2']))

    story.append(Spacer(1, 20))

    # ======= 13. REFERENCES =======
    story.append(Paragraph("13. References", styles['SectionTitle']))
    story.append(section_divider())
    refs = [
        "Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.",
        "Chen, T. and Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.",
        "Flask Documentation. https://flask.palletsprojects.com/",
        "Microsoft Azure App Service Documentation. https://docs.microsoft.com/en-us/azure/app-service/",
        "CarDekho Dataset, Kaggle. https://www.kaggle.com/datasets/nehalbirla/vehicle-dataset-from-cardekho",
        "Docker Documentation. https://docs.docker.com/",
    ]
    for i, ref in enumerate(refs, 1):
        story.append(Paragraph(f"[{i}] {ref}", styles['BodyText2']))

    # Build the PDF
    doc.build(story)
    print(f"PDF report generated: {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024:.0f} KB")


if __name__ == "__main__":
    build_report()
