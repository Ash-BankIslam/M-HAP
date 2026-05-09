import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import re
from mbb_logic import run_mbb_resampling
from xgb_logic import train_and_evaluate

st.set_page_config(page_title="UiTM Weather AI", layout="wide")

# Helper function to create Professional Gauge Charts
def create_gauge(value, title, max_range, custom_steps=None):
    steps = custom_steps if custom_steps else [
        {'range': [0, max_range * 0.33], 'color': "#81C784"}, 
        {'range': [max_range * 0.33, max_range * 0.66], 'color': "#FFD54F"}, 
        {'range': [max_range * 0.66, max_range], 'color': "#E57373"}  
    ]
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = value,
        number = {'font': {'size': 44, 'color': "#2C3E50", 'family': "Arial, sans-serif"}},
        title = {'text': f"<span style='font-size: 22px; color: #555555; font-weight: bold; font-family: Arial, sans-serif'>{title}</span>"},
        gauge = {
            'axis': {'range': [None, max_range], 'tickwidth': 1, 'tickcolor': "#BDC3C7", 'tickfont': {'color': "#7F8C8D"}},
            'bar': {'color': "#2C3E50", 'thickness': 0.15},
            'bgcolor': "#F2F3F4",
            'borderwidth': 0,
            'steps': steps,
        },
        # THE FIX: Moves the gauge down slightly within the container (0 is bottom, 1 is top)
        domain = {'x': [0, 1], 'y': [0, 0.85]} 
    ))
    
    fig.update_layout(
        height=250, # Slightly increased height for better spacing
        # THE FIX: Increased 't' (top margin) from 40 to 80
        margin=dict(l=20, r=20, t=80, b=10), 
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig

# Custom CSS for the banners
def banner(text):
    st.markdown(f"""
        <div style='background-color: #8A1538; color: white; padding: 8px; 
                    text-align: center; font-size: 20px; font-weight: bold; 
                    margin-top: 20px; margin-bottom: 20px; border: 1px solid black;'>
            {text}
        </div>
        """, unsafe_allow_html=True)

# --- TOP HEADER (Vertically Centered) ---
_, col_center_img, _ = st.columns([1, 1, 1])
with col_center_img:
    st.image("UiTM FSKM.png", width=350)

st.markdown("<h1 style='text-align: center;'>☁️ PM10 Forecast Hub</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Monitor current environmental conditions and 24-hour particulate matter predictions.</p>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Hourly Data", type=['xlsx', 'csv', 'xls'])

if uploaded_file:
    # Bulletproof file reading
    try:
        if uploaded_file.name.endswith(('.xlsx', '.xls')):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        uploaded_file.seek(0)
        try:
            df = pd.read_csv(uploaded_file, sep='\t')
        except Exception as inner_e:
            st.error(f"Could not read file. Error: {inner_e}")
            st.stop()
            
    # --- SMART DATE PARSER ---
    if 'Datetime' not in df.columns and 'Date' in df.columns:
        if pd.api.types.is_numeric_dtype(df['Date']):
            df['Datetime'] = pd.to_datetime(df['Date'].astype(str), format='%Y%m%d', errors='coerce')
        else:
            df['Datetime'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Get the latest row for the dashboard gauges
    latest = df.iloc[-1]
    
    # --- DYNAMIC STATION NAME EXTRACTION ---
    raw_name = uploaded_file.name
    clean_name = os.path.splitext(raw_name)[0]
    words_to_remove = ['hourly', 'complete', 'data', 'dataset', 'csv', 'excel', 'final']
    for word in words_to_remove:
        clean_name = re.sub(rf'\b{word}\b', '', clean_name, flags=re.IGNORECASE)
        
    station_name = clean_name.replace('_', ' ').replace('-', ' ').strip().upper()
    if not station_name:
        station_name = "UNKNOWN STATION"
    
    # ==========================================
    # --- EDA SECTION ---
    # ==========================================
    col_station, col_blank = st.columns([1, 4])
    col_station.markdown(f"<div style='background-color: #8A1538; color: white; padding: 5px; text-align: center; font-weight: bold; border: 1px solid black;'>STATION: {station_name}</div>", unsafe_allow_html=True)
    st.write("") 

    col_gauge, col_info = st.columns([1, 2])
    pm10_val = latest.get('PM10', latest.get('PM10,24HOUR', 0))
    
    with col_gauge:
        st.markdown("<div style='background-color: #8A1538; color: white; padding: 5px; text-align: center; font-weight: bold; border: 1px solid black;'>AIR STATUS</div>", unsafe_allow_html=True)
        pm10_steps = [
            {'range': [0, 100], 'color': "#81C784"},     # Muted Green (Healthy)
            {'range': [100, 150], 'color': "#FFD54F"},   # Soft Yellow (Moderate)
            {'range': [150, 200], 'color': "#FFB74D"},   # Soft Orange (Unhealthy)
            {'range': [200, 300], 'color': "#E57373"}    # Muted Red (Extreme)
        ]
        st.plotly_chart(create_gauge(pm10_val, "PM10", max_range=300, custom_steps=pm10_steps), use_container_width=True)
        
    with col_info:
        if pm10_val < 100:
            status, color = "Healthy", "#D4EFDF"
            actions = "<li>Enjoy normal outdoor activities.</li><li>Ideal for opening windows for ventilation.</li>"
        elif 100 <= pm10_val < 150:
            status, color = "Moderate", "#FCF3CF"
            actions = "<li>Consider reducing prolonged or heavy exercise.</li><li>Watch for symptoms such as coughing or shortness of breath.</li>"
        elif 150 <= pm10_val < 200:
            status, color = "Unhealthy", "#FDEBD0"
            actions = "<li>Avoid prolonged or heavy exertion.</li><li>More activities indoor or reschedule to a time when air quality is better.</li>"
        else:
            status, color = "Extreme / Very Unhealthy", "#FADBD8"
            actions = "<li>Avoid all physical activities.</li><li>Remain indoors and keep activity levels low.</li>"
            
        st.markdown(f"""
        <div style='background-color: {color}; padding: 20px; border-radius: 5px; height: 180px; margin-top: 35px;'>
            <h4>Air Status: {status}</h4>
            <p><b>Suggested Action:</b></p>
            <ul>{actions}</ul>
        </div>
        """, unsafe_allow_html=True)

    # 2. POLLUTANT LEVELS SECTION
    banner("POLLUTANT LEVELS")
    p1, p2, p3, p4, p5 = st.columns(5)
    
    # ULTIMATE FAIL-SAFE: This function intercepts NaN, missing cells, and empty spaces
    def clean_val(col_name):
        val = latest.get(col_name, 0)
        # If it's NaN or an empty space, force it to be 0.0
        if pd.isna(val) or str(val).strip() == "":
            return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0

    # Extract safely using our new function
    val_so2 = clean_val('SO2')
    val_no2 = clean_val('NO2')
    val_co  = clean_val('CO')
    val_o3  = clean_val('O3')
    val_nox = clean_val('NOX')
    
    with p1: st.plotly_chart(create_gauge(val_so2, "SO₂", 200, [{'range': [0, 30], 'color': "#81C784"}, {'range': [30, 95], 'color': "#FFD54F"}, {'range': [95, 200], 'color': "#E57373"}]), use_container_width=True)
    with p2: st.plotly_chart(create_gauge(val_no2, "NO₂", 300, [{'range': [0, 37], 'color': "#81C784"}, {'range': [37, 149], 'color': "#FFD54F"}, {'range': [149, 300], 'color': "#E57373"}]), use_container_width=True)
    with p3: st.plotly_chart(create_gauge(val_co, "CO", 40000, [{'range': [0, 8700], 'color': "#81C784"}, {'range': [8700, 26200], 'color': "#FFD54F"}, {'range': [26200, 40000], 'color': "#E57373"}]), use_container_width=True)
    with p4: st.plotly_chart(create_gauge(val_o3, "O₃", 150, [{'range': [0, 51], 'color': "#81C784"}, {'range': [51, 92], 'color': "#FFD54F"}, {'range': [92, 150], 'color': "#E57373"}]), use_container_width=True)
    with p5: st.plotly_chart(create_gauge(val_nox, "NOX", 300, [{'range': [0, 100], 'color': "#81C784"}, {'range': [100, 170], 'color': "#FFD54F"}, {'range': [170, 300], 'color': "#E57373"}]), use_container_width=True)

    # 3. METEOROLOGICAL CONDITIONS SECTION
    banner("METEOROLOGICAL CONDITIONS")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ TEMPERATURE", f"{latest.get('Ambient Temperature', 0)} °C")
    m2.metric("💧 HUMIDITY", f"{latest.get('Relative Humidity', 0)} %")
    m3.metric("💨 WIND SPEED", f"{latest.get('Wind Speed', 0)}")
    m4.metric("🧭 WIND DIRECTION", f"{latest.get('Wind Direction Index', 0)}")

    # ==========================================
    # --- HISTORICAL TREND GRAPH SECTION ---
    # ==========================================
    banner("PM10 HISTORICAL & PERIODIC TRENDS")
    
    if 'Datetime' in df.columns and not df['Datetime'].isna().all():
        target_col = 'PM10,24HOUR' if 'PM10,24HOUR' in df.columns else 'PM10'
        
        st.subheader("📅 1. Long-Term Yearly Overview")
        overall_avg_pm10 = df[target_col].mean()
        
        if 'Year' not in df.columns:
             df['Year'] = df['Datetime'].dt.year
        yearly_avg_df = df.groupby('Year')[target_col].mean().reset_index()
        
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; margin-bottom: 20px; background-color: #F8F9F9; border-left: 5px solid #8A1538; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='margin:0; color: #333;'>Global Historical Average (PM10)</h3>
            <h1 style='margin:0; color: #8A1538; font-size: 48px;'>{overall_avg_pm10:.2f}</h1>
        </div>
        """, unsafe_allow_html=True)

        fig_yearly = px.line(yearly_avg_df, x='Year', y=target_col, markers=True, title="Average PM10 by Year")
        fig_yearly.add_hline(y=100, line_dash="dash", line_color="#FFEA00", annotation_text="Moderate Limit")
        fig_yearly.add_hline(y=150, line_dash="dash", line_color="#FFB300", annotation_text="Unhealthy Limit")
        fig_yearly.add_hline(y=overall_avg_pm10, line_dash="dot", line_color="blue", annotation_text=f"Overall Avg ({overall_avg_pm10:.1f})", annotation_position="bottom right")
        fig_yearly.update_layout(xaxis_title="Year", yaxis_title="Average PM10 Level", hovermode="x unified", xaxis=dict(tickmode='linear', dtick=1), margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_yearly, use_container_width=True)

        st.divider()

        st.subheader("🔍 2. Interactive Periodic Analysis")
        period = st.radio("Select Time Period", ["7 Days", "1 Month", "3 Months", "6 Months", "1 Year", "Overall"], index=4, horizontal=True, label_visibility="collapsed")
        
        max_date = df['Datetime'].max()
        if period == "7 Days": start_date = max_date - pd.Timedelta(days=7)
        elif period == "1 Month": start_date = max_date - pd.DateOffset(months=1)
        elif period == "3 Months": start_date = max_date - pd.DateOffset(months=3)
        elif period == "6 Months": start_date = max_date - pd.DateOffset(months=6)
        elif period == "1 Year": start_date = max_date - pd.DateOffset(years=1)
        else: start_date = df['Datetime'].min()
            
        filtered_df = df[df['Datetime'] >= start_date]
        
        fig_period = px.line(filtered_df, x='Datetime', y=target_col, title=f"PM10 Timeline ({period})")
        fig_period.add_hline(y=100, line_dash="dash", line_color="#FFEA00", annotation_text="Moderate Limit (100)")
        fig_period.add_hline(y=150, line_dash="dash", line_color="#FFB300", annotation_text="Unhealthy Limit (150)")
        fig_period.add_hline(y=200, line_dash="dash", line_color="#FF1744", annotation_text="Extreme Limit (200)")
        fig_period.update_layout(xaxis_title="Timeline", yaxis_title="PM10 Level", hovermode="x unified", margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_period, use_container_width=True)
        
    else:
        st.warning("Could not detect a valid Date column to generate the trend graph. Ensure you have a 'Date' column.")

    # ==========================================
    # --- RESAMPLING & MODELING SECTION ---
    # ==========================================
    banner("PREDICTIVE PIPELINE & SIMULATOR")
    
    st.subheader("🚀 1. Automated Pipeline (Resample & Train)")
    st.markdown("This will apply **MBB-Ratio Resampling** to balance extreme events, and immediately train the **XGBoost Engine**.")
    
    thresh = st.slider("Extreme Threshold", 50, 200, 150) 
    
    # --- THE ONE-CLICK PIPELINE BUTTON ---
    if st.button("Run Pipeline (Resample & Train)", type="primary"):
        
        # Step 1: Resampling
        with st.spinner("Step 1: Running MBB Resampling to balance data..."):
            balanced, msg = run_mbb_resampling(df, threshold=thresh)
            
        if balanced is not None:
            st.session_state['data'] = balanced
            st.success(msg)
            
            # Step 2: Training
            with st.spinner("Step 2: Training XGBoost Engine on balanced data..."):
                model, metrics, X_test, y_test, y_pred = train_and_evaluate(balanced)
                
                # Save everything to session state
                st.session_state['xgb_model'] = model
                st.session_state['xgb_metrics'] = metrics
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
                
            st.success("✅ Model successfully trained and ready!")
        else:
            st.error(msg)

    # --- DISPLAY RESULTS & SIMULATOR ---
    # Check if the pipeline has successfully finished and saved data in memory
    model_ready = all(k in st.session_state for k in ['xgb_model', 'xgb_metrics', 'y_test', 'y_pred'])

    if model_ready:
        st.divider()
        st.subheader("🤖 2. XGBoost Prediction Engine")
        
        # Retrieve model data from state
        model = st.session_state['xgb_model']
        metrics = st.session_state['xgb_metrics']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        feature_names = model.get_booster().feature_names
        latest_row = df.iloc[-1]
        
        # ------------------------------------------
        # MODEL METRICS & CHART 
        # ------------------------------------------
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.write("**Model Evaluation Metrics**")
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
        with res_col2:
            # 1. Create the interactive Zoom Buttons
            zoom_option = st.radio(
                "🔍 Select Zoom Level (Data Points):",
                options=["Last 250", "Last 500", "Last 1000", "Overall"],
                horizontal=True,
                index=0 # Defaults to 250 so it loads cleanly
            )
            
            # 2. Determine the slice size mathematically
            if zoom_option == "Last 250": slice_idx = -250
            elif zoom_option == "Last 500": slice_idx = -500
            elif zoom_option == "Last 1000": slice_idx = -1000
            else: slice_idx = 0 # 0 means grab everything
            
            # 3. Slice the arrays based on the user's choice
            if slice_idx < 0:
                y_test_sliced = np.array(y_test)[slice_idx:]
                y_pred_sliced = y_pred[slice_idx:]
            else:
                y_test_sliced = np.array(y_test)
                y_pred_sliced = y_pred

            # 4. Build the dataframe for the chart
            comparison_df = pd.DataFrame({
                'Sample Sequence': range(len(y_test_sliced)),
                'Actual T+24 (PM10,24HOUR)': y_test_sliced,
                'Predicted T+24': y_pred_sliced
            })
            
            # 5. Plot the dynamic graph
            fig_res = px.line(
                comparison_df, 
                x='Sample Sequence', 
                y=['Actual T+24 (PM10,24HOUR)', 'Predicted T+24'],
                labels={'value': 'PM10 Concentration', 'variable': 'Legend'}, 
                title=f"Actual vs Predicted Forecast ({zoom_option})",
                color_discrete_map={'Actual T+24 (PM10,24HOUR)': '#1F77B4', 'Predicted T+24': '#D62728'}
            )
            
            # Smart Styling: If they choose 'Overall', make the lines thinner so it's less messy
            line_width = 1 if zoom_option == "Overall" else 2
            fig_res.update_traces(line=dict(width=line_width))
            
            fig_res.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig_res, use_container_width=True)

        # ------------------------------------------
        # AUTOMATED LATEST PREDICTION KPI 
        # ------------------------------------------
        st.divider()
        
        kpi_df = pd.DataFrame({col: [0.0] for col in feature_names})
        
        for col in feature_names:
            if col in latest_row.index:
                kpi_df.at[0, col] = float(latest_row[col])
            col_upper = str(col).upper()
            if 'PM10' in col_upper and '24HOUR' not in col_upper: kpi_df.at[0, col] = float(latest_row.get('PM10', 0))
            elif 'SPEED' in col_upper: kpi_df.at[0, col] = float(latest_row.get('Wind Speed', 0))
            elif 'DIR' in col_upper: kpi_df.at[0, col] = float(latest_row.get('Wind Direction Index', 0))
            elif 'TEMP' in col_upper: kpi_df.at[0, col] = float(latest_row.get('Ambient Temperature', 0))
            elif 'HUMIDITY' in col_upper or 'RH' in col_upper: kpi_df.at[0, col] = float(latest_row.get('Relative Humidity', 0))
            elif 'NOX' in col_upper: kpi_df.at[0, col] = float(latest_row.get('NOX', 0))
            elif 'SO2' in col_upper: kpi_df.at[0, col] = float(latest_row.get('SO2', 0))
            elif 'NO2' in col_upper: kpi_df.at[0, col] = float(latest_row.get('NO2', 0))
            elif 'O3' in col_upper: kpi_df.at[0, col] = float(latest_row.get('O3', 0))
            elif 'CO' in col_upper: kpi_df.at[0, col] = float(latest_row.get('CO', 0))
        
        latest_pred = model.predict(kpi_df.values)[0]
        
        current_pm10 = float(latest_row.get('PM10', 0))
        actual_tomorrow = float(latest_row.get('PM10,24HOUR', 0))
        
        diff_val = latest_pred - current_pm10
        diff_pct = (diff_val / current_pm10) * 100 if current_pm10 != 0 else 0
        
        if latest_pred < 100: pred_status, pred_color = "🟢 Healthy", "#D4EFDF"
        elif 100 <= latest_pred < 150: pred_status, pred_color = "🟡 Moderate", "#FCF3CF"
        elif 150 <= latest_pred < 200: pred_status, pred_color = "🟠 Unhealthy", "#FDEBD0"
        else: pred_status, pred_color = "🔴 Extreme", "#FADBD8"
            
        arrow = "🔺" if diff_val > 0 else "🔻"
        delta_color = "red" if diff_val > 0 else "green"
        
        st.markdown(f"""
        <div style='background-color: {pred_color}; padding: 25px; border-radius: 10px; border: 2px solid #ccc; text-align: center; margin-bottom: 30px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);'>
            <h3 style='margin:0; color: #333; text-transform: uppercase;'>Latest PM10 Forecast (Next 24H)</h3>
            <h1 style='margin:0; font-size: 64px; color: #111;'>{latest_pred:.1f}</h1>
            <h2 style='margin:0; margin-bottom: 15px;'>Expected Status: {pred_status}</h2>
            <div style='background-color: white; display: inline-block; padding: 10px 20px; border-radius: 20px; font-weight: bold; font-size: 18px; color: {delta_color}; margin-bottom: 10px;'>
                {arrow} {abs(diff_val):.1f} ({abs(diff_pct):.1f}%) change from Current PM10 ({current_pm10:.1f})
            </div>
            <p style='margin:0; font-size: 14px; color: #555;'><i>Actual target recorded in dataset (PM10,24HOUR): <b>{actual_tomorrow:.1f}</b></i></p>
        </div>
        """, unsafe_allow_html=True)

        # ------------------------------------------
        # WHAT-IF SIMULATOR 
        # ------------------------------------------
        st.divider()
        banner("WHAT-IF SIMULATOR")
        st.markdown("Adjust the environmental parameters below to simulate how the model predicts the PM10 24-Hour average will react. *Note: Hourly PM10 is included for reference, but the model primarily uses gases and weather to forecast tomorrow's air.*")
        
        with st.form("simulator_form"):
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            
            sim_pm10 = col_sim1.number_input("Hourly PM10", value=float(latest.get('PM10', 0)))
            sim_so2 = col_sim1.number_input("SO2", value=float(latest.get('SO2', 0)))
            sim_no2 = col_sim1.number_input("NO2", value=float(latest.get('NO2', 0)))
            sim_co = col_sim1.number_input("CO", value=float(latest.get('CO', 0)))
            
            sim_o3 = col_sim2.number_input("O3", value=float(latest.get('O3', 0)))
            sim_nox = col_sim2.number_input("NOX", value=float(latest.get('NOX', 0)))
            
            sim_temp = col_sim3.number_input("Ambient Temperature (°C)", value=float(latest.get('Ambient Temperature', 0)))
            sim_rh = col_sim3.number_input("Relative Humidity (%)", value=float(latest.get('Relative Humidity', 0)))
            sim_ws = col_sim3.number_input("Wind Speed", value=float(latest.get('Wind Speed', 0)))
            sim_wd = col_sim3.number_input("Wind Direction Index", value=float(latest.get('Wind Direction Index', 0)))
            
            submitted = st.form_submit_button("Run Simulation", type="primary")
            
            if submitted:
                sim_df = pd.DataFrame({col: [0.0] for col in feature_names})
                
                for col in feature_names:
                    if col in latest_row.index:
                        sim_df.at[0, col] = float(latest_row[col])
                        
                for col in feature_names:
                    col_upper = str(col).upper()
                    if 'PM10' in col_upper and '24HOUR' not in col_upper: sim_df.at[0, col] = float(sim_pm10)
                    elif 'SPEED' in col_upper: sim_df.at[0, col] = float(sim_ws)
                    elif 'DIR' in col_upper: sim_df.at[0, col] = float(sim_wd)
                    elif 'TEMP' in col_upper: sim_df.at[0, col] = float(sim_temp)
                    elif 'HUMIDITY' in col_upper or 'RH' in col_upper: sim_df.at[0, col] = float(sim_rh)
                    elif 'NOX' in col_upper: sim_df.at[0, col] = float(sim_nox)
                    elif 'SO2' in col_upper: sim_df.at[0, col] = float(sim_so2)
                    elif 'NO2' in col_upper: sim_df.at[0, col] = float(sim_no2)
                    elif 'O3' in col_upper: sim_df.at[0, col] = float(sim_o3)
                    elif 'CO' in col_upper: sim_df.at[0, col] = float(sim_co)
                
                sim_pred = model.predict(sim_df.values)[0]
                
                if sim_pred < 100: s_status, s_color = "Healthy", "#D4EFDF"
                elif 100 <= sim_pred < 150: s_status, s_color = "Moderate", "#FCF3CF"
                elif 150 <= sim_pred < 200: s_status, s_color = "Unhealthy", "#FDEBD0"
                else: s_status, s_color = "Extreme", "#FADBD8"

                st.markdown(f"""
                <div style='background-color: {s_color}; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #333; margin-top: 20px;'>
                    <h4 style='margin:0;'>Simulated Forecast (Next 24H)</h4>
                    <h2 style='margin:0; font-size: 42px;'>{sim_pred:.1f}</h2>
                    <h5 style='margin:0;'>Status: {s_status}</h5>
                </div>
                """, unsafe_allow_html=True)