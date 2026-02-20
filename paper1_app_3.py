#!/usr/bin/env python
# coding: utf-8

# In[ ]:



import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="GCD Analyzer", layout="centered")


# -----------------------------
# 🔐 LOGIN SCREEN
# -----------------------------
def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if not st.session_state.password_correct:
        st.title("Login")
        password = st.text_input("Enter password", type="password")

        if password == "Battery":
            st.session_state.password_correct = True
            st.rerun()
        elif password != "":
            st.error("Incorrect password")

        st.stop()


check_password()


# -----------------------------
# MAIN APP
# -----------------------------
st.title("Dissipative Effects – GCD Analysis Tool")

uploaded_file = st.file_uploader("Upload CSV or Excel file", type=["csv", "xlsx"])

col1, col2 = st.columns(2)

with col1:
    I_app = st.number_input("Applied current (A)", value=0.0005, format="%.6f")
    mass_g = st.number_input("Active mass (g)", value=0.0004, format="%.6f")

with col2:
    device_label = st.selectbox("Device type", ["Supercapacitor", "Battery"])
    time_col = st.text_input("Time column name", "Elapsed Time (s)")
    voltage_col = st.text_input("Voltage column name", "Voltage(V)")

device_type = "SC" if device_label == "Supercapacitor" else "Bat"


# -----------------------------
# LOAD DATA
# -----------------------------
def load_data(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    else:
        return pd.read_excel(file)


# -----------------------------
# COMPUTATION
# -----------------------------
def compute_all(t, U):

    discharge_time = t[-1] - t[0]
    U_max = U[0]
    U_min = U[-1]

    A_real = np.trapz(U, t)

    if device_type == "SC":
        A_ideal = U_max * discharge_time / 2
    else:
        A_ideal = U_max * discharge_time

    gamma = A_real / A_ideal

    E_real_J = I_app * A_real
    E_ideal_J = I_app * A_ideal

    mass_kg = mass_g / 1000

    E_real_spec = (E_real_J / 3600) / mass_kg
    E_ideal_spec = (E_ideal_J / 3600) / mass_kg

    discharge_time_h = discharge_time / 3600
    P_real_spec = E_real_spec / discharge_time_h

    return gamma, A_real, A_ideal, E_real_spec, E_ideal_spec, P_real_spec, discharge_time, U_max


# -----------------------------
# PLOT
# -----------------------------
def plot_energy(t, U, A_real, A_ideal, discharge_time, U_max):

    fig, ax = plt.subplots()

    ax.plot(t, U, color="black", linewidth=2)

    # Real energy
    ax.fill_between(t, U, color="#cce5ff", label="E real")

    # Ideal triangle
    t_ideal = np.linspace(0, discharge_time, 500)
    U_ideal = U_max * (1 - t_ideal / discharge_time)
    ax.fill_between(t_ideal, U_ideal, color="#d4edda", label="E ideal")

    # Gamma loss
    U_interp = np.interp(t_ideal, t, U)
    ax.fill_between(
        t_ideal,
        U_interp,
        U_ideal,
        where=(U_ideal > U_interp),
        color="#ffe5cc",
        label="γ loss",
    )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    st.pyplot(fig)


# -----------------------------
# RUN
# -----------------------------
if uploaded_file is not None:

    df = load_data(uploaded_file)

    if st.button("Run analysis"):

        t = df[time_col].values
        U = df[voltage_col].values

        gamma, A_real, A_ideal, E_real_spec, E_ideal_spec, P_real_spec, discharge_time, U_max = compute_all(t, U)

        st.subheader("Results")

        st.write("γ =", gamma)

        st.write("Real specific energy:", E_real_spec, "Wh/kg")
        st.write("Ideal specific energy:", E_ideal_spec, "Wh/kg")
        st.write("Corrected ⟨E⟩:", gamma * E_ideal_spec, "Wh/kg")

        st.write("Real specific power:", P_real_spec, "W/kg")

        st.subheader("Energy visualization")

        plot_energy(t, U, A_real, A_ideal, discharge_time, U_max)

