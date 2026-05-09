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
        domain = {'x': [0, 1], 'y': [0, 0.85]} 
    ))
    
    fig.update_layout(
        height=250,
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

# --- TOP HEADER ---
_, col_center_img, _ = st.columns([1, 1, 1])
with col_center_img:
    st.image("UiTM FSKM.png", width=350)

st.markdown("<h1 style='text-align: center;'>☁️ M-HAP: Malaysian Haze Analysis and Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Monitor current environmental conditions and 24-hour particulate matter predictions.</p>", unsafe_allow_html=True)

st.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Hourly Data", type=['xlsx', 'csv', 'xls'])

if uploaded_file:
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
    
    # --- EDA SECTION ---
    col_station, col_blank = st.columns([1, 4])
    col_station.markdown(f"<div style='background-color: #8A1538; color: white; padding: 5px; text-align: center; font-weight: bold; border: 1px solid black;'>STATION: {station_name}</div>", unsafe_allow_html=True)
    st.write("") 

    col_gauge, col_info = st.columns([1, 2])
    pm10_val = latest.get('PM10', latest.get('PM10,24HOUR', 0))
    
    with col_gauge:
        st.markdown("<div style='background-color: #8A1538; color: white; padding: 5px; text-align: center; font-weight: bold; border: 1px solid black;'>AIR STATUS</div>", unsafe_allow_html=True)
        pm10_steps = [
            {'range': [0, 100], 'color': "#81C784"},
            {'range': [100, 150], 'color': "#FFD54F"},
            {'range': [150, 200], 'color': "#FFB74D"},
            {'range': [200, 300], 'color': "#E57373"}
        ]
        # UPDATED UNIT: PM10 (µg/m³)
        st.plotly_chart(create_gauge(pm10_val, "PM₁₀ (µg/m³)", max_range=300, custom_steps=pm10_steps), use_container_width=True)
        
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
    
    def clean_val(col_name):
        val = latest.get(col_name, 0)
        if pd.isna(val) or str(val).strip() == "":
            return 0.0
        try:
            return float(val)
        except ValueError:
            return 0.0

    val_so2 = clean_val('SO2')
    val_no2 = clean_val('NO2')
    val_co  = clean_val('CO')
    val_o3  = clean_val('O3')
    val_nox = clean_val('NOX')
    
    with p1: st.plotly_chart(create_gauge(val_so2, "SO₂ (ppb)", 200, [{'range': [0, 30], 'color': "#81C784"}, {'range': [30, 95], 'color': "#FFD54F"}, {'range': [95, 200], 'color': "#E57373"}]), use_container_width=True)
    with p2: st.plotly_chart(create_gauge(val_no2, "NO₂ (ppb)", 300, [{'range': [0, 37], 'color': "#81C784"}, {'range': [37, 149], 'color': "#FFD54F"}, {'range': [149, 300], 'color': "#E57373"}]), use_container_width=True)
    with p3: st.plotly_chart(create_gauge(val_co, "CO (ppb)", 40000, [{'range': [0, 8700], 'color': "#81C784"}, {'range': [8700, 26200], 'color': "#FFD54F"}, {'range': [26200, 40000], 'color': "#E57373"}]), use_container_width=True)
    with p4: st.plotly_chart(create_gauge(val_o3, "O₃ (ppb)", 150, [{'range': [0, 51], 'color': "#81C784"}, {'range': [51, 92], 'color': "#FFD54F"}, {'range': [92, 150], 'color': "#E57373"}]), use_container_width=True)
    with p5: st.plotly_chart(create_gauge(val_nox, "NOx (ppb)", 300, [{'range': [0, 100], 'color': "#81C784"}, {'range': [100, 170], 'color': "#FFD54F"}, {'range': [170, 300], 'color': "#E57373"}]), use_container_width=True)

    # 3. METEOROLOGICAL CONDITIONS SECTION
    banner("METEOROLOGICAL CONDITIONS")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🌡️ TEMPERATURE (°C)", f"{latest.get('Ambient Temperature', 0)} °C")
    m2.metric("💧 HUMIDITY (%)", f"{latest.get('Relative Humidity', 0)} %")
    m3.metric("💨 WIND SPEED (m/s)", f"{latest.get('Wind Speed', 0)} m/s")
    m4.metric("🧭 WIND DIRECTION (°)", f"{latest.get('Wind Direction Index', 0)} °")

    # --- HISTORICAL TREND GRAPH SECTION ---
    banner("PM₁₀ HISTORICAL & PERIODIC TRENDS")
    
    if 'Datetime' in df.columns and not df['Datetime'].isna().all():
        target_col = 'PM10,24HOUR' if 'PM10,24HOUR' in df.columns else 'PM10'
        
        st.subheader("📅 1. Long-Term Yearly Overview")
        overall_avg_pm10 = df[target_col].mean()
        
        if 'Year' not in df.columns:
             df['Year'] = df['Datetime'].dt.year
        yearly_avg_df = df.groupby('Year')[target_col].mean().reset_index()
        
        st.markdown(f"""
        <div style='text-align: center; padding: 15px; margin-bottom: 20px; background-color: #F8F9F9; border-left: 5px solid #8A1538; border-radius: 5px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1);'>
            <h3 style='margin:0; color: #333;'>Global Historical Average (PM₁₀)</h3>
            <h1 style='margin:0; color: #8A1538; font-size: 48px;'>{overall_avg_pm10:.2f} <span style='font-size: 20px;'>µg/m³</span></h1>
        </div>
        """, unsafe_allow_html=True)

        fig_yearly = px.line(yearly_avg_df, x='Year', y=target_col, markers=True, title="Average PM₁₀ by Year")
        fig_yearly.add_hline(y=100, line_dash="dash", line_color="#FFEA00", annotation_text="Moderate Limit")
        fig_yearly.add_hline(y=150, line_dash="dash", line_color="#FFB300", annotation_text="Unhealthy Limit")
        fig_yearly.update_layout(xaxis_title="Year", yaxis_title="Avg PM₁₀ Level (µg/m³)", hovermode="x unified", xaxis=dict(tickmode='linear', dtick=1))
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
        
        fig_period = px.line(filtered_df, x='Datetime', y=target_col, title=f"PM₁₀ Timeline ({period})")
        fig_period.update_layout(xaxis_title="Timeline", yaxis_title="PM₁₀ Level (µg/m³)", hovermode="x unified")
        st.plotly_chart(fig_period, use_container_width=True)
        
    else:
        st.warning("Could not detect a valid Date column.")

    # --- RESAMPLING & MODELING SECTION ---
    banner("PREDICTIVE PIPELINE & SIMULATOR")
    
    st.subheader("🚀 1. Automated Pipeline (Resample & Train)")
    thresh = st.slider("Extreme Threshold", 50, 200, 150) 
    
    if st.button("Run Pipeline (Resample & Train)", type="primary"):
        with st.spinner("Step 1: Running MBB Resampling..."):
            balanced, msg = run_mbb_resampling(df, threshold=thresh)
            
        if balanced is not None:
            st.session_state['data'] = balanced
            with st.spinner("Step 2: Training XGBoost..."):
                model, metrics, X_test, y_test, y_pred = train_and_evaluate(balanced)
                st.session_state['xgb_model'] = model
                st.session_state['xgb_metrics'] = metrics
                st.session_state['y_test'] = y_test
                st.session_state['y_pred'] = y_pred
            st.success("✅ Model successfully trained and ready!")
        else:
            st.error(msg)

    model_ready = all(k in st.session_state for k in ['xgb_model', 'xgb_metrics', 'y_test', 'y_pred'])

    if model_ready:
        st.divider()
        st.subheader("🤖 2. XGBoost Prediction Engine")
        
        model = st.session_state['xgb_model']
        metrics = st.session_state['xgb_metrics']
        y_test = st.session_state['y_test']
        y_pred = st.session_state['y_pred']
        feature_names = model.get_booster().feature_names
        latest_row = df.iloc[-1]
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.write("**Model Evaluation Metrics**")
            metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
            metrics_df['Value'] = metrics_df['Value'].apply(lambda x: f"{x:.4f}")
            st.dataframe(metrics_df, hide_index=True, use_container_width=True)
            
        with res_col2:
            zoom_option = st.radio("🔍 Select Zoom Level:", options=["Last 250", "Last 500", "Last 1000", "Overall"], horizontal=True)
            slice_idx = -250 if zoom_option == "Last 250" else -500 if zoom_option == "Last 500" else -1000 if zoom_option == "Last 1000" else 0
            
            y_test_sliced = np.array(y_test)[slice_idx:] if slice_idx < 0 else np.array(y_test)
            y_pred_sliced = y_pred[slice_idx:] if slice_idx < 0 else y_pred

            comparison_df = pd.DataFrame({'Sample Sequence': range(len(y_test_sliced)), 'Actual': y_test_sliced, 'Predicted': y_pred_sliced})
            fig_res = px.line(comparison_df, x='Sample Sequence', y=['Actual', 'Predicted'], title=f"Actual vs Predicted Forecast ({zoom_option})",color_discrete_map={'Actual': '#1F77B4', 'Predicted': '#FF0000'})
            st.plotly_chart(fig_res, use_container_width=True)

        st.divider()
        
        # Prepare feature vector for prediction
        kpi_df = pd.DataFrame({col: [0.0] for col in feature_names})
        for col in feature_names:
            col_upper = str(col).upper()
            if col in latest_row.index: kpi_df.at[0, col] = float(latest_row[col])
            # Logic for matching feature names to data columns
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
        
        if latest_pred < 100: pred_status, pred_color, rec_bg, rec_text = "🟢 Healthy", "#D4EFDF", "#E8F8F5", "<li>Air quality is satisfactory.</li><li>Good time for outdoor exercise and ventilation.</li>"
        elif 100 <= latest_pred < 150: pred_status, pred_color, rec_bg, rec_text = "🟡 Moderate", "#FCF3CF", "#FEF9E7", "<li>Sensitive individuals should reduce heavy outdoor exertion.</li>"
        elif 150 <= latest_pred < 200: pred_status, pred_color, rec_bg, rec_text = "🟠 Unhealthy", "#FDEBD0", "#FEF5E7", "<li>Limit prolonged outdoor exposure. Close windows.</li>"
        else: pred_status, pred_color, rec_bg, rec_text = "🔴 Extreme", "#FADBD8", "#FDEDEC", "<li><b>STAY INDOORS.</b> Avoid all physical activities outside.</li>"

        col_pred, col_rec = st.columns([1.5, 2.5])
        with col_pred:
            st.markdown(f"""
            <div style='background-color: {pred_color}; padding: 20px; border-radius: 10px; border: 2px solid #ccc; text-align: center; height: 350px;'>
                <h4 style='margin:0;'>FORECAST (24H)</h4>
                <h1 style='margin:10px 0; font-size: 58px;'>{latest_pred:.1f}</h1>
                <h3 style='margin:0;'>{pred_status}</h3>
                <p style='margin-top:15px; font-size: 14px;'><i>Unit: µg/m³</i></p>
            </div>
            """, unsafe_allow_html=True)
        with col_rec:
            st.markdown(f"""
            <div style='background-color: {rec_bg}; padding: 25px; border-radius: 10px; border: 1px solid {pred_color}; height: 350px;'>
                <h3>📋 Suggested Precautions</h3><hr>
                <ul style='font-size: 18px; line-height: 1.6;'>{rec_text}</ul>
            </div>
            """, unsafe_allow_html=True)

        # --- WHAT-IF SIMULATOR ---
        st.divider()
        banner("WHAT-IF SIMULATOR")
        st.markdown("Adjust parameters to see how the predicted **PM₁₀** reacts.")
        
        with st.form("simulator_form"):
            col_sim1, col_sim2, col_sim3 = st.columns(3)
            sim_pm10 = col_sim1.number_input("Hourly PM₁₀ (µg/m³)", value=float(latest.get('PM10', 0)))
            sim_so2 = col_sim1.number_input("SO2 (ppb)", value=float(latest.get('SO2', 0)))
            sim_no2 = col_sim1.number_input("NO2 (ppb)", value=float(latest.get('NO2', 0)))
            sim_co = col_sim1.number_input("CO (ppb)", value=float(latest.get('CO', 0)))
            sim_o3 = col_sim2.number_input("O3 (ppb)", value=float(latest.get('O3', 0)))
            sim_nox = col_sim2.number_input("NOX (ppb)", value=float(latest.get('NOX', 0)))
            sim_temp = col_sim3.number_input("Temp (°C)", value=float(latest.get('Ambient Temperature', 0)))
            sim_rh = col_sim3.number_input("Humidity (%)", value=float(latest.get('Relative Humidity', 0)))
            sim_ws = col_sim3.number_input("Wind Speed (m/s)", value=float(latest.get('Wind Speed', 0)))
            sim_wd = col_sim3.number_input("Wind Direction (°)", value=float(latest.get('Wind Direction Index', 0)))
            submitted = st.form_submit_button("Run Simulation", type="primary")
            
            if submitted:
                sim_df = pd.DataFrame({col: [0.0] for col in feature_names})
                for col in feature_names:
                    col_upper = str(col).upper()
                    if col in latest_row.index: sim_df.at[0, col] = float(latest_row[col])
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
                
                # Determine simulation status and precaution
                if sim_pred < 100: s_status, s_color, s_rec = "Healthy", "#D4EFDF", "Air quality is satisfactory. No precautions needed."
                elif 100 <= sim_pred < 150: s_status, s_color, s_rec = "Moderate", "#FCF3CF", "Sensitive groups should reduce heavy outdoor exertion."
                elif 150 <= sim_pred < 200: s_status, s_color, s_rec = "Unhealthy", "#FDEBD0", "Limit prolonged outdoor exposure; keep windows closed."
                else: s_status, s_color, s_rec = "Extreme", "#FADBD8", "STAY INDOORS. Avoid all physical activities outside."

                sim_res_col, sim_rec_col = st.columns([1, 1])
                with sim_res_col:
                    st.markdown(f"""
                    <div style='background-color: {s_color}; padding: 15px; border-radius: 5px; text-align: center; border: 1px solid #333; height: 150px;'>
                        <h4 style='margin:0; font-size: 14px;'>Simulated Forecast (24H)</h4>
                        <h2 style='margin:5px 0; font-size: 38px;'>{sim_pred:.1f} <span style='font-size: 14px;'>µg/m³</span></h2>
                        <h5 style='margin:0;'>Status: {s_status}</h5>
                    </div>
                    """, unsafe_allow_html=True)
                with sim_rec_col:
                    st.markdown(f"""
                    <div style='background-color: #F8F9F9; padding: 15px; border-radius: 5px; border-left: 5px solid {s_color}; height: 150px;'>
                        <h4 style='margin:0; font-size: 14px; color: #555;'>SIMULATED PRECAUTION</h4>
                        <p style='margin-top: 10px; font-size: 15px; font-weight: bold; color: #2C3E50;'>{s_rec}</p>
                    </div>
                    """, unsafe_allow_html=True)