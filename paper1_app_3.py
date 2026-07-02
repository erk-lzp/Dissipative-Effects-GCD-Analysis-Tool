"""
GCD Analyzer
------------
A small Streamlit app for analyzing galvanostatic charge-discharge (GCD)
curves. It works for both supercapacitors and batteries, computes the
"gamma" shape factor of the discharge curve, and reports energy/power
densities. It also draws an energy-region plot and a Ragone diagram.

Run with:  streamlit run gcd_analyzer.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


st.set_page_config(page_title="GCD Analyzer", layout="centered")

# A single place to change the password if needed, we used while we were testing
# APP_PASSWORD = "Battery"


# Login 

#def require_login():
    """Ask for a password before showing anything else.

    We keep a flag in session_state so the user only has to log in once
    per session instead of on every rerun.
    """
    # First time we run, nobody is logged in yet.
 #   if "logged_in" not in st.session_state:
     #   st.session_state.logged_in = False

  #  if st.session_state.logged_in:
    #    return

   # st.title("Login")
    #entered = st.text_input("Enter password", type="password")

    #if entered == APP_PASSWORD:
     #   st.session_state.logged_in = True
     #   st.rerun()
    #elif entered:
        # Only complain once the user has actually typed something.
      #  st.error("Incorrect password")

    # Stop here until the password is correct.
    #st.stop()


# File loading

def load_data(uploaded_file):
    """Read an uploaded CSV or Excel file into a DataFrame."""
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


# Labels and units 
def get_units(device_type, normalization_basis):
    """Pick the right names and units for the chosen device / basis.

    Supercapacitors and mass-normalized batteries are reported per kg.
    Volume-normalized batteries are reported per liter of electrolyte.
    """
    per_volume = (
        device_type == "Battery"
        and normalization_basis == "Electrolyte volume"
    )

    if per_volume:
        return {
            "energy_name": "Volumetric energy density",
            "power_name": "Volumetric power density",
            "energy_unit": "Wh/dm3",
            "power_unit": "W/dm3",
        }

    # Default: gravimetric (per mass).
    return {
        "energy_name": "Specific energy",
        "power_name": "Specific power",
        "energy_unit": "Wh/kg",
        "power_unit": "W/kg",
    }


# --- Core calculation ------------------------------------------------------

def calculate_metrics(t, U, current_A, device_type, normalization_basis,
                      active_mass_g=None, electrolyte_volume_dm3=None):
    """Compute gamma, energy and power from a discharge curve.

    The idea: compare the real area under the voltage-time curve to an
    "ideal" reference area. For a supercapacitor the ideal discharge is a
    straight line (triangle); for a battery it's a flat plateau (rectangle).
    Gamma is just the ratio of the two areas, so it tells you how close the
    real curve is to that ideal shape.

    Parameters
    ----------
    t, U : array-like
        Time (s) and voltage (V) samples of the discharge.
    current_A : float
        Applied discharge current in amperes.
    device_type : str
        "Supercapacitor" or "Battery".
    normalization_basis : str
        "Active mass" or "Electrolyte volume" (batteries only).
    active_mass_g, electrolyte_volume_dm3 : float
        Whichever one matches the chosen normalization basis.

    Returns
    -------
    dict of computed values.
    """
    # Total discharge duration and the voltage window.
    discharge_time = t[-1] - t[0]
    U_start = U[0]
    U_end = U[-1]

    # Real area under the measured curve (trapezoidal integration).
    area_real = np.trapz(U, t)

    # Ideal reference area depends on the expected discharge shape.
    if device_type == "Supercapacitor":
        # Linear ramp down -> triangle.
        area_ideal = U_start * discharge_time / 2
    else:
        # Flat plateau -> rectangle.
        area_ideal = U_start * discharge_time

    # Shape factor: 1.0 means the curve matches the ideal exactly.
    gamma = area_real / area_ideal

    # Energy in joules:  E = I * integral(U dt).
    energy_real_J = current_A * area_real
    energy_ideal_J = current_A * area_ideal

    # Decide what we divide by to normalize (mass in kg, or volume in dm3).
    if device_type == "Supercapacitor" or normalization_basis == "Active mass":
        norm_factor = active_mass_g / 1000.0       # grams -> kg
    else:
        norm_factor = electrolyte_volume_dm3        # already in dm3

    # Convert joules to watt-hours, then normalize.
    energy_real = (energy_real_J / 3600) / norm_factor
    energy_ideal = (energy_ideal_J / 3600) / norm_factor
    energy_corrected = gamma * energy_ideal

    # Average power = energy / time (with time in hours).
    discharge_time_h = discharge_time / 3600
    power_real = energy_real / discharge_time_h

    return {
        "gamma": gamma,
        "area_real": area_real,
        "area_ideal": area_ideal,
        "energy_real": energy_real,
        "energy_ideal": energy_ideal,
        "energy_corrected": energy_corrected,
        "power_real": power_real,
        "discharge_time": discharge_time,
        "U_start": U_start,
        "U_end": U_end,
    }


# --- Plots -----------------------------------------------------------------

def plot_energy(t, U, device_type, discharge_time, U_start):
    """Show the measured curve along with the ideal and 'lost' energy areas."""
    fig, ax = plt.subplots(figsize=(7, 5))

    # The actual measured discharge.
    ax.plot(t, U, color="black", linewidth=2, label="Experimental curve")

    # Everything under the real curve is real (delivered) energy.
    ax.fill_between(t, U, color="#cce5ff", alpha=0.8, label="Real energy")

    # Build the ideal curve over the same time span.
    t_ideal = np.linspace(0, discharge_time, 500)
    if device_type == "Supercapacitor":
        U_ideal = U_start * (1 - t_ideal / discharge_time)   # triangle
    else:
        U_ideal = np.full_like(t_ideal, U_start)             # rectangle

    ax.fill_between(t_ideal, U_ideal, color="#d4edda", alpha=0.5,
                    label="Ideal energy")

    # The gap between ideal and real is the energy "lost" to non-ideality.
    U_on_ideal_grid = np.interp(t_ideal, t, U)
    ax.fill_between(t_ideal, U_on_ideal_grid, U_ideal,
                    where=(U_ideal > U_on_ideal_grid),
                    color="#ffe5cc", alpha=0.8, label="gamma loss")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(alpha=0.3)
    ax.legend()
    st.pyplot(fig)


def plot_ragone(energy_value, power_value, energy_unit, power_unit, device_name):
    """Plot a single point on a log-log Ragone diagram (energy vs power)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(energy_value, power_value, s=120, marker="o")
    ax.text(energy_value, power_value, device_name, fontsize=10)

    # Ragone plots are conventionally log-log.
    ax.set_xscale("log")
    ax.set_yscale("log")

    ax.set_xlabel(f"Energy density ({energy_unit})")
    ax.set_ylabel(f"Power density ({power_unit})")
    ax.set_title("Ragone Plot")
    ax.grid(which="both", alpha=0.3)
    st.pyplot(fig)


# --- Results display -------------------------------------------------------

def display_results(results, units):
    """Print the numeric results in a readable block."""
    st.subheader("Results")

    st.write(f"gamma = {results['gamma']:.4f}")
    st.write(f"Real {units['energy_name'].lower()}: "
             f"{results['energy_real']:.4f} {units['energy_unit']}")
    st.write(f"Ideal {units['energy_name'].lower()}: "
             f"{results['energy_ideal']:.4f} {units['energy_unit']}")
    st.write(f"Corrected energy (gamma x E_ideal): "
             f"{results['energy_corrected']:.4f} {units['energy_unit']}")
    st.write(f"Real {units['power_name'].lower()}: "
             f"{results['power_real']:.4f} {units['power_unit']}")


def generate_plots(t, U, results, units, device_type):
    """Draw both figures one after the other."""
    st.subheader("Energy visualization")
    plot_energy(
        t=t,
        U=U,
        device_type=device_type,
        discharge_time=results["discharge_time"],
        U_start=results["U_start"],
    )

    st.subheader("Ragone plot")
    plot_ragone(
        energy_value=results["energy_real"],
        power_value=results["power_real"],
        energy_unit=units["energy_unit"],
        power_unit=units["power_unit"],
        device_name=device_type,
    )


# --- Input helpers ---------------------------------------------------------

def collect_basic_inputs():
    """Two-column block for current, device type and column names."""
    col1, col2 = st.columns(2)

    with col1:
        current_A = st.number_input("Applied current (A)",
                                    value=0.0005, format="%.6f")
        device_type = st.selectbox("Device type",
                                   ["Supercapacitor", "Battery"])

    with col2:
        time_col = st.text_input("Time column name", "Elapsed Time (s)")
        voltage_col = st.text_input("Voltage column name", "Voltage(V)")

    return current_A, device_type, time_col, voltage_col


def collect_normalization_inputs(device_type):
    """Ask for mass or electrolyte volume depending on the device.

    Returns (normalization_basis, active_mass_g, electrolyte_volume_dm3),
    with the unused value left as None.
    """
    st.subheader("Normalization")

    normalization_basis = "Active mass"
    active_mass_g = None
    electrolyte_volume_dm3 = None

    if device_type == "Supercapacitor":
        active_mass_g = st.number_input("Active mass (g)",
                                        value=0.0004, format="%.6f")
    else:
        normalization_basis = st.selectbox(
            "Normalization basis",
            ["Active mass", "Electrolyte volume"],
        )
        if normalization_basis == "Active mass":
            active_mass_g = st.number_input("Active mass (g)",
                                            value=0.0004, format="%.6f")
        else:
            # Let the user pick the volume unit they actually measured in.
            # The mL field is per pole (each electrode / half-cell), which
            # is usually how flow-battery electrolyte is dispensed.
            volume_unit = st.selectbox(
                "Electrolyte volume unit",
                ["dm3 (total)", "mL (each pole / electrode)"],
            )

            if volume_unit.startswith("mL"):
                volume_ml_per_pole = st.number_input(
                    "Electrolyte volume per pole (mL)",
                    value=800.0, format="%.2f")
                # mL -> dm3 (/1000), then x2 because there are two poles
                # (anolyte + catholyte) and densities are reported on the
                # total electrolyte volume.
                electrolyte_volume_dm3 = (volume_ml_per_pole / 1000.0) * 2
                st.caption(
                    f"Total electrolyte volume used for normalization: "
                    f"{electrolyte_volume_dm3:.4f} dm3"
                )
            else:
                electrolyte_volume_dm3 = st.number_input(
                    "Electrolyte volume (dm3, total)",
                    value=0.0100, format="%.4f")

    return normalization_basis, active_mass_g, electrolyte_volume_dm3


# --- Main ------------------------------------------------------------------

def main():
    st.title("GCD-\u03b3 Analyzer: A tool for the precise evaluation of energy and power characteristics in electrochemical energy storage devices")
    #st.caption("A tool for energy correction in electrochemical "
     #          "energy storage devices")

    uploaded_file = st.file_uploader("Upload CSV or Excel file",
                                     type=["csv", "xlsx"])

    # Gather all settings up front so the layout stays stable.
    current_A, device_type, time_col, voltage_col = collect_basic_inputs()
    norm_basis, active_mass_g, electrolyte_volume_dm3 = \
        collect_normalization_inputs(device_type)

    # Nothing to do until a file is provided.
    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to begin.")
        return

    # Try to read the file and show a quick preview.
    try:
        df = load_data(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df.head())
    except Exception as err:
        st.error(f"Error reading file:\n{err}")
        return

    # The heavy work only runs when the user clicks the button.
    if not st.button("Run analysis"):
        return

    # Pull out the two columns we actually need.
    try:
        t = df[time_col].to_numpy()
        U = df[voltage_col].to_numpy()
    except KeyError:
        st.error("Column names not found. "
                 "Please verify the time and voltage columns.")
        return

    results = calculate_metrics(
        t=t,
        U=U,
        current_A=current_A,
        device_type=device_type,
        normalization_basis=norm_basis,
        active_mass_g=active_mass_g,
        electrolyte_volume_dm3=electrolyte_volume_dm3,
    )

    units = get_units(device_type, norm_basis)

    display_results(results, units)
    generate_plots(t, U, results, units, device_type)


if __name__ == "__main__":
    require_login()
    main()
