# Houston Astros Analytics Dashboard

An interactive dashboard built with Dash to analyze Houston Astros roster data and performance metrics.

## Features

- WAR Analysis by age, position, and season
- Career longevity insights
- Position-based analysis
- Player demographics visualization
- Interactive filters and controls
- Dark theme with Houston Astros color scheme

## Installation

1. Clone the repository:
```bash
git clone https://github.com/MikeWayne92/diamond.git
cd diamond
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements-core.txt
```

## Usage

1. Run the server:
```bash
python run_server.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:8050
```

## Data

The dashboard uses Houston Astros roster data stored in `Astros.data/Houston Astros Roster Data.csv`. Make sure this file is present before running the application.

## Development

To contribute to this project:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Data source: [Kaggle](https://www.kaggle.com/datasets/dasanson/boston-red-sox-roster-data-1965-2020)
- Built with [Plotly Dash](https://dash.plotly.com/)
- Color scheme based on official Houston Astros colors# diamond
