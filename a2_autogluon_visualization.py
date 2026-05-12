import pandas as pd
import numpy as np
import os
import webbrowser
from autogluon.tabular import TabularPredictor
from sklearn.metrics import precision_score, recall_score, matthews_corrcoef

# Plotting and Dashboarding
import plotly.express as px
from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc

class CustomEvaluator:
    def __init__(self):
        # 1. DATA LOADING
        # ---------------------------------------------------------
        X = pd.read_csv(rf'D:\Option Data 2\unscaled_features\X_1.csv')
        y = pd.read_csv(rf'D:\Option Data 2\unscaled_features\y_1.csv')
        X['Date'] = pd.to_datetime(X['Date'])
        
        y['Profitable Trade'] = y['PNL'] >= 0.15
        Xy = X.copy()
        Xy['Profitable Trade'] = y['Profitable Trade'].values
        Xy['PNL'] = y['PNL'].values

        # Validation Split
        self.validation_df = Xy[(Xy['Date'] >= '2021-01-01') & (Xy['Date'] <= '2022-12-31')].copy()
        self.validation_df['Market Cap'] = np.exp(self.validation_df['Log Market Cap'])
        
        # FIXED UNIVERSE QUANTILES (0-19)
        self.validation_df['MCap_Bin'] = pd.qcut(self.validation_df['Market Cap'], 20, labels=False, duplicates='drop')

        # 2. MODEL PREDICTIONS
        # ---------------------------------------------------------
        model_path = rf"C:\Users\I'm the best\Documents\a\Earnings Estimation\Thesis_2\AutogluonModels\precision_mid"
        predictor = TabularPredictor.load(model_path, require_py_version_match=False)
        
        label = "Profitable Trade"
        X_val_data = self.validation_df.drop(columns=[label, 'Date', 'Q-String', 'PNL', 'Market Cap', 'MCap_Bin'], errors='ignore')
        y_true = self.validation_df[label]
        
        results = []
        self.trade_details = [] 
        years_in_sample = 2

        for model_name in predictor.model_names():
            y_pred = predictor.predict(X_val_data, model=model_name)
            trades_mask = (y_pred == 1)
            
            model_trades = self.validation_df.loc[trades_mask].copy()
            model_trades['Model'] = model_name
            self.trade_details.append(model_trades)

            trade_returns = self.validation_df.loc[trades_mask, 'PNL']
            
            # CALCULATIONS
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec = recall_score(y_true, y_pred, zero_division=0)
            avg_ret = trade_returns.mean() if not trade_returns.empty else 0
            tot_ret = trade_returns.sum() if not trade_returns.empty else 0
            std = trade_returns.std()
            
            # Annualized Sharpe
            sharpe = (avg_ret / std * np.sqrt(trades_mask.sum() / years_in_sample)) if (not trade_returns.empty and std != 0) else 0

            results.append({
                'Model': model_name, 
                'Precision': round(prec, 4), 
                'Recall': round(rec, 4),
                'Trades': int(trades_mask.sum()), 
                'Avg Return': round(avg_ret, 4),
                'Total Return': round(tot_ret, 4),
                'Sharpe Ratio': round(sharpe, 4)
            })

        self.df_results = pd.DataFrame(results).sort_values(by='Sharpe Ratio', ascending=False)
        self.all_trades_df = pd.concat(self.trade_details)
        self.baseline_pct = y_true.mean()

        # 3. PRE-GENERATE STATIC UNIVERSE PLOTS
        self.fig_full_universe_scatter = px.scatter(
            self.validation_df, x='Market Cap', y='PNL', color='Profitable Trade',
            color_discrete_map={True: '#00CC96', False: '#EF553B'}, template='plotly_dark', opacity=0.3,
            title="Universe Context: Raw Returns"
        )
        
        universe_sum = self.validation_df.groupby('MCap_Bin')['PNL'].sum().reset_index()
        self.fig_universe_sum_bar = px.bar(
            universe_sum, x='MCap_Bin', y='PNL', color='PNL',
            color_continuous_scale='RdYlGn', template='plotly_dark',
            title="Universe Total Profit Pool (Sum per Quantile)"
        )

        self.run_dashboard()

    def run_dashboard(self):
        app = Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

        # Updated PR Plot with extra metrics in hover
        fig_main = px.scatter(
            self.df_results, x='Precision', y='Recall', color='Model',
            hover_name='Model', 
            custom_data=['Model'],
            hover_data={
                'Precision': ':.3f',
                'Recall': ':.3f',
                'Avg Return': ':.4f',
                'Total Return': ':.2f',
                'Sharpe Ratio': ':.3f',
                'Trades': True,
                'Model': False
            },
            title=f'Model Comparison (Baseline: {self.baseline_pct:.2%})', 
            template='plotly_dark'
        )
        fig_main.update_traces(marker=dict(size=14, line=dict(width=1, color='White')))

        app.layout = dbc.Container([
            dbc.Row([dbc.Col(html.H2("Model Performance & Market Cap Analysis", className="text-center my-4"))]),
            
            # PR Curve (Top)
            dbc.Row([dbc.Col(dcc.Graph(id='pr-scatter', figure=fig_main))]),
            
            # Model Specific Analysis (Middle)
            dbc.Row([
                dbc.Col([
                    html.Hr(),
                    html.Div(id='model-header', className="text-center py-2"),
                    html.H5("Model-Specific Trades (Individual Returns)", className="text-muted"),
                    dcc.Graph(id='trade-scatter-plot'),
                    html.H5("Model Profit Pool (Sum per Fixed Universe Quantile)", className="text-muted"),
                    dcc.Graph(id='model-sum-bar')
                ], width=12)
            ]),

            # Universe Baseline (Bottom)
            dbc.Row([
                dbc.Col([
                    html.Hr(style={'borderColor': '#555'}),
                    html.H3("Universe Baseline (All Opportunities)", className="text-center py-2"),
                    dcc.Graph(figure=self.fig_full_universe_scatter),
                    dcc.Graph(figure=self.fig_universe_sum_bar)
                ], width=12)
            ])
        ], fluid=True)

        @app.callback(
            [Output('trade-scatter-plot', 'figure'), 
             Output('model-sum-bar', 'figure'),
             Output('model-header', 'children')],
            [Input('pr-scatter', 'clickData')]
        )
        def update_model_plots(clickData):
            if clickData is None:
                empty = px.scatter(title="Click a model on the PR Curve", template='plotly_dark')
                return empty, empty, html.H4("No Model Selected")
            
            selected_model = clickData['points'][0]['customdata'][0]
            df_model = self.all_trades_df[self.all_trades_df['Model'] == selected_model]

            # Model Specific Scatter
            fig_scat = px.scatter(
                df_model, x='Market Cap', y='PNL', color='PNL',
                color_continuous_scale='RdYlGn', template='plotly_dark',
                hover_data=['Date', 'Q-String']
            )
            fig_scat.add_hline(y=0, line_dash="dash", line_color="gray")

            # Model Specific Sum per FIXED quantile
            model_sum = df_model.groupby('MCap_Bin')['PNL'].sum().reset_index()
            fig_bar = px.bar(
                model_sum, x='MCap_Bin', y='PNL', color='PNL',
                color_continuous_scale='RdYlGn', template='plotly_dark',
                labels={'MCap_Bin': 'Fixed Market Cap Quantile', 'PNL': 'Cumulative Model PNL'}
            )
            fig_bar.update_layout(xaxis=dict(tickmode='linear', tick0=0, dtick=1))

            header = html.H3(f"Deep Dive: {selected_model}", style={'color': '#00CC96'})
            return fig_scat, fig_bar, header

        port = 8090
        webbrowser.open(f"http://127.0.0.1:{port}")
        app.run(debug=False, port=port)

if __name__ == "__main__":
    evaluator = CustomEvaluator()