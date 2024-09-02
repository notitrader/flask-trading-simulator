from flask import Flask, render_template, request, send_file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get input values from the form
        contracts = int(request.form.get("contracts", 1))
        min_ticks_profit = int(request.form.get("min_ticks_profit", 3))
        max_ticks_profit = int(request.form.get("max_ticks_profit", 7))
        ticks_loss = int(request.form.get("ticks_loss", 5))
        tick_value = float(request.form.get("tick_value", 12.5))
        fee_per_contract = float(request.form.get("fee_per_contract", 2.5))
        num_trades = int(request.form.get("num_trades", 200))
        breakeven_trades = int(request.form.get("breakeven_trades", 10)) / 100
        win_rate = int(request.form.get("win_rate", 60)) / 100

        if min_ticks_profit >= max_ticks_profit:
            return render_template('index.html', error="Minimum Profit Ticks must be less than Maximum Profit Ticks.")

        adjusted_win_rate = win_rate * (1 - breakeven_trades)
        loss_rate = 1 - adjusted_win_rate - breakeven_trades
        num_variations = int(request.form.get("num_variations", 10))

        # Simulation
        simulation_results = {}
        ticks_used = {}

        for variation in range(1, num_variations + 1):
            profits = []
            ticks = []
            for _ in range(num_trades):
                random_value = np.random.rand()
                if random_value <= breakeven_trades:
                    profit = -(fee_per_contract * contracts * 2)  # Only fees paid on opening and closing
                    ticks.append(0)
                elif random_value <= breakeven_trades + adjusted_win_rate:
                    random_ticks_profit = np.random.randint(min_ticks_profit, max_ticks_profit + 1)
                    profit = (random_ticks_profit * tick_value * contracts) - (fee_per_contract * contracts * 2)  # Winning trade minus fees
                    ticks.append(random_ticks_profit)
                else:
                    profit = -(ticks_loss * tick_value * contracts) - (fee_per_contract * contracts * 2)  # Losing trade plus fees
                    ticks.append(-ticks_loss)
                profits.append(profit)
            cumulative_profit = np.cumsum(profits)
            simulation_results[f'Variation {variation}'] = cumulative_profit
            ticks_used[f'Variation {variation}'] = ticks

        # Creating DataFrame to display results
        df_simulation = pd.DataFrame(simulation_results)
        df_ticks = pd.DataFrame(ticks_used)

        # Replace NaN, inf, and -inf with 0 for display purposes
        df_simulation.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
        df_ticks.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

        # Combine profits and ticks into a single DataFrame with a MultiIndex
        combined_data = {}
        for variation in range(1, num_variations + 1):
            combined_data[(f'Variation {variation}', 'Profits')] = df_simulation[f'Variation {variation}'].apply(lambda x: f"${x:,.2f}")
            combined_data[(f'Variation {variation}', 'Ticks')] = df_ticks[f'Variation {variation}']

        df_combined = pd.DataFrame(combined_data)

        # Metrics calculation
        selected_variation = request.form.get("selected_variation", "All Variations")
        if selected_variation == "All Variations":
            selected_variations = df_simulation.columns.tolist()
            avg_profit = df_simulation[selected_variations].iloc[-1].mean()
            drawdown = df_simulation[selected_variations].cummax() - df_simulation[selected_variations]
            max_drawdown = drawdown.max().max()
            sharpe_ratio = (df_simulation[selected_variations].mean().mean() / df_simulation[selected_variations].std().mean()) * np.sqrt(252)
        else:
            selected_variations = [selected_variation]
            avg_profit = df_simulation[selected_variation].iloc[-1].mean()
            drawdown = df_simulation[selected_variation].cummax() - df_simulation[selected_variation]
            max_drawdown = drawdown.max().max()
            sharpe_ratio = (df_simulation[selected_variation].mean() / df_simulation[selected_variation].std()) * np.sqrt(252)

        # Convert DataFrame to HTML table
        df_combined_html = df_combined.to_html(classes='table table-striped', index=False)

        return render_template('index.html', df_combined_html=df_combined_html, avg_profit=avg_profit, max_drawdown=max_drawdown, sharpe_ratio=sharpe_ratio)

    return render_template('index.html')

@app.route('/download', methods=['POST'])
def download():
    csv = request.form['csv_data']
    b = io.BytesIO()
    b.write(csv.encode())
    b.seek(0)
    return send_file(b, as_attachment=True, download_name='simulation_results.csv', mimetype='text/csv')

if __name__ == '__main__':
    app.run(debug=True)
