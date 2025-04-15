import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import warnings
import sys
from pathlib import Path
import importlib

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('default')  # Using default style instead of seaborn
sns.set_theme()  # This is the modern way to set seaborn style

def check_dependencies():
    """Check if all required packages are installed and importable"""
    required_packages = {
        'pandas': 'pd',
        'numpy': 'np',
        'matplotlib.pyplot': 'plt',
        'seaborn': 'sns',
        'scipy': 'stats',
        'plotly.express': 'px',
        'sklearn': 'sklearn'
    }
    
    missing_packages = []
    for package, alias in required_packages.items():
        try:
            importlib.import_module(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Error: Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        sys.exit(1)

def verify_data_file():
    """Verify that the data file exists and is readable"""
    data_file = Path('Astros.data/Houston Astros Roster Data.csv')
    if not data_file.exists():
        print(f"Error: Data file not found at {data_file}")
        print("Please ensure the data file is in the correct location.")
        sys.exit(1)
    if not data_file.is_file():
        print(f"Error: {data_file} is not a file")
        sys.exit(1)
    try:
        with open(data_file, 'r') as f:
            first_line = f.readline()
        if not first_line:
            print(f"Error: {data_file} is empty")
            sys.exit(1)
    except Exception as e:
        print(f"Error reading data file: {str(e)}")
        sys.exit(1)

def setup_output_directory():
    """Set up the output directory"""
    output_dir = Path('output')
    try:
        output_dir.mkdir(exist_ok=True)
        # Test if we can write to the directory
        test_file = output_dir / 'test.txt'
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        print(f"Error setting up output directory: {str(e)}")
        sys.exit(1)
    return output_dir

def load_data():
    """Load and preprocess the data with error handling"""
    try:
        file_path = Path('Astros.data/Houston Astros Roster Data.csv')
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found at {file_path}")
        
        # Read CSV with specific data types
        df = pd.read_csv(file_path)
        
        # Convert height to inches with error handling
        def convert_height(height_str):
            try:
                if pd.isna(height_str):
                    return np.nan
                feet, inches = height_str.split("'")
                return float(feet) * 12 + float(inches.strip('"'))
            except (ValueError, AttributeError):
                return np.nan
        
        # Convert numeric columns
        numeric_columns = ['Age', 'Wt', 'Yrs', 'GP', 'GS', 'WAR']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Convert height to inches
        df['Height_inches'] = df['Ht'].apply(convert_height)
        
        # Convert DoB to datetime with error handling
        df['DoB'] = pd.to_datetime(df['DoB'], errors='coerce')
        
        # Convert position columns to numeric
        position_cols = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF', 'OF', 'DH']
        for col in position_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Create positions_played column
        df['positions_played'] = df[position_cols].apply(lambda x: sum(x > 0), axis=1)
        
        # Convert binary columns
        df['All-Star'] = (df['All-Star'] == 'Yes').astype(int)
        df['HOF'] = (df['HOF'] == 'Yes').astype(int)
        
        # Handle missing values
        df = df.fillna({
            'WAR': 0,
            'GP': 0,
            'GS': 0,
            'positions_played': 0,
            'Height_inches': df['Height_inches'].mean(),
            'Wt': df['Wt'].mean(),
            'Age': df['Age'].mean()
        })
        
        return df
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise

# Load data
df = load_data()

# 1. WAR Analysis
def analyze_war():
    try:
        plt.figure(figsize=(15, 10))
        
        # WAR vs Age
        plt.subplot(2, 2, 1)
        sns.lineplot(data=df, x='Age', y='WAR')
        plt.title('WAR vs Age')
        
        # WAR by Position (for primary position)
        position_war = []
        position_labels = []
        for pos in ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']:
            pos_data = df[df[pos] > 0]['WAR']
            if len(pos_data) > 0:
                position_war.append(pos_data)
                position_labels.append(pos)
        
        plt.subplot(2, 2, 2)
        if position_war:  # Only plot if we have data
            plt.boxplot(position_war, labels=position_labels)
            plt.title('WAR Distribution by Position')
            plt.xticks(rotation=45)
        
        # WAR Trends Over Seasons
        plt.subplot(2, 2, 3)
        season_war = df.groupby('Season')['WAR'].mean().reset_index()
        sns.lineplot(data=season_war, x='Season', y='WAR')
        plt.title('Average WAR by Season')
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'war_analysis.png')
        plt.close()
    except Exception as e:
        print(f"Error in WAR analysis: {str(e)}")
        raise

# 2. Career Longevity Analysis
def analyze_career_longevity():
    try:
        plt.figure(figsize=(15, 10))
        
        # Career Length Distribution
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='Yrs', bins=int(df['Yrs'].max()))
        plt.title('Distribution of Career Lengths')
        
        # Average WAR by Age
        plt.subplot(2, 2, 2)
        age_war = df.groupby('Age')['WAR'].mean().reset_index()
        sns.lineplot(data=age_war, x='Age', y='WAR')
        plt.title('Average WAR by Age')
        
        # Player Retention Rate
        max_years = int(df['Yrs'].max())
        retention = [(df['Yrs'] >= yr).mean() for yr in range(1, max_years + 1)]
        
        plt.subplot(2, 2, 3)
        plt.plot(range(1, max_years + 1), retention)
        plt.title('Player Retention Rate')
        plt.xlabel('Years in League')
        plt.ylabel('Proportion of Players Remaining')
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'career_longevity.png')
        plt.close()
    except Exception as e:
        print(f"Error in career longevity analysis: {str(e)}")
        raise

# 3. Position Versatility Analysis
def analyze_position_versatility():
    try:
        plt.figure(figsize=(15, 10))
        
        # Number of Positions Distribution
        plt.subplot(2, 2, 1)
        max_positions = int(df['positions_played'].max())
        sns.histplot(data=df, x='positions_played', bins=range(0, max_positions + 2))
        plt.title('Distribution of Positions Played')
        
        # Position Combinations
        position_cols = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        position_counts = df[position_cols].apply(lambda x: x > 0).sum()
        
        plt.subplot(2, 2, 2)
        position_counts.plot(kind='bar')
        plt.title('Number of Players by Position')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'position_versatility.png')
        plt.close()
    except Exception as e:
        print(f"Error in position versatility analysis: {str(e)}")
        raise

# 4. Demographics Analysis
def analyze_demographics():
    try:
        plt.figure(figsize=(15, 10))
        
        # Player Origins
        plt.subplot(2, 2, 1)
        country_counts = df['Born'].value_counts()
        country_counts.plot(kind='bar')
        plt.title('Player Origins')
        plt.xticks(rotation=45)
        
        # WAR by Country
        plt.subplot(2, 2, 2)
        country_war = df.groupby('Born')['WAR'].mean().sort_values(ascending=False)
        country_war.plot(kind='bar')
        plt.title('Average WAR by Country of Origin')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'demographics.png')
        plt.close()
    except Exception as e:
        print(f"Error in demographics analysis: {str(e)}")
        raise

# 5. All-Star & HOF Analysis
def analyze_allstar_hof():
    try:
        # Prepare features for prediction
        features = ['Age', 'WAR', 'GP', 'positions_played', 'Height_inches', 'Wt']
        X = df[features].fillna(0)  # Ensure no missing values
        y_allstar = df['All-Star'].astype(int)
        
        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train simple logistic regression
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_allstar, test_size=0.2, random_state=42)
        model = LogisticRegression(random_state=42)
        model.fit(X_train, y_train)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        importance = pd.DataFrame({
            'feature': features,
            'importance': abs(model.coef_[0])
        }).sort_values('importance', ascending=True)
        
        plt.barh(importance['feature'], importance['importance'])
        plt.title('Feature Importance for All-Star Prediction')
        plt.xlabel('Absolute Coefficient Value')
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'allstar_prediction.png')
        plt.close()
    except Exception as e:
        print(f"Error in All-Star & HOF analysis: {str(e)}")
        raise

# 6. Handedness Analysis
def analyze_handedness():
    try:
        plt.figure(figsize=(15, 10))
        
        # Batting Hand Distribution
        plt.subplot(2, 2, 1)
        batting_counts = df['B'].value_counts()
        plt.pie(batting_counts.values, labels=batting_counts.index, autopct='%1.1f%%')
        plt.title('Batting Hand Distribution')
        
        # Throwing Hand Distribution
        plt.subplot(2, 2, 2)
        throwing_counts = df['T'].value_counts()
        plt.pie(throwing_counts.values, labels=throwing_counts.index, autopct='%1.1f%%')
        plt.title('Throwing Hand Distribution')
        
        # WAR by Handedness
        plt.subplot(2, 2, 3)
        sns.boxplot(data=df, x='B', y='WAR')
        plt.title('WAR Distribution by Batting Hand')
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'handedness.png')
        plt.close()
    except Exception as e:
        print(f"Error in handedness analysis: {str(e)}")
        raise

# 7. Physical Attributes Analysis
def analyze_physical_attributes():
    try:
        plt.figure(figsize=(15, 10))
        
        # Height vs Weight Scatter
        plt.subplot(2, 2, 1)
        sns.scatterplot(data=df, x='Height_inches', y='Wt', hue='WAR', size='WAR')
        plt.title('Height vs Weight (colored by WAR)')
        
        # Height Distribution by Position
        plt.subplot(2, 2, 2)
        position_cols = ['P', 'C', '1B', '2B', '3B', 'SS', 'LF', 'CF', 'RF']
        height_by_pos = []
        pos_labels = []
        for pos in position_cols:
            pos_height = df[df[pos] > 0]['Height_inches']
            if len(pos_height) > 0:
                height_by_pos.append(pos_height)
                pos_labels.append(pos)
        
        if height_by_pos:  # Only plot if we have data
            plt.boxplot(height_by_pos, labels=pos_labels)
            plt.title('Height Distribution by Position')
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(Path('output') / 'physical_attributes.png')
        plt.close()
    except Exception as e:
        print(f"Error in physical attributes analysis: {str(e)}")
        raise

def create_output_directory():
    """Create output directory for plots if it doesn't exist"""
    output_dir = Path('output')
    output_dir.mkdir(exist_ok=True)
    return output_dir

def save_plot(fig, filename, output_dir):
    """Save plot with error handling"""
    try:
        plt.savefig(output_dir / filename)
        plt.close()
    except Exception as e:
        print(f"Error saving plot {filename}: {str(e)}")

def run_all_analyses():
    """Run all analyses with error handling"""
    try:
        # Create output directory
        output_dir = create_output_directory()
        
        # Run analyses
        analyze_war()
        analyze_career_longevity()
        analyze_position_versatility()
        analyze_demographics()
        analyze_allstar_hof()
        analyze_handedness()
        analyze_physical_attributes()
        
        # Generate summary statistics
        summary_stats = {
            'Total Players': len(df['Name'].unique()),
            'Years Covered': f"{df['Season'].min()} - {df['Season'].max()}",
            'Average WAR': round(df['WAR'].mean(), 2),
            'All-Stars': (df['All-Star'] == 'Yes').sum(),
            'Hall of Famers': (df['HOF'] == 'Yes').sum(),
            'Average Career Length': round(df['Yrs'].mean(), 2)
        }
        
        # Save summary statistics
        try:
            with open(output_dir / 'analysis_summary.txt', 'w') as f:
                f.write("Houston Astros Historical Analysis Summary\n")
                f.write("=======================================\n\n")
                for key, value in summary_stats.items():
                    f.write(f"{key}: {value}\n")
        except Exception as e:
            print(f"Error writing summary file: {str(e)}")
        
        print("Analysis complete. Check the output directory for results.")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        print("Checking dependencies...")
        check_dependencies()
        
        print("Verifying data file...")
        verify_data_file()
        
        print("Setting up output directory...")
        output_dir = setup_output_directory()
        
        print("Loading data...")
        df = load_data()
        
        print("Running analyses...")
        run_all_analyses()
        
        print("Analysis completed successfully!")
        print("Results have been saved to the 'output' directory.")
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1) 