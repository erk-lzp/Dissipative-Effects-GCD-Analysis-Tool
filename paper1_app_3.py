"""GCD Analyzer
Streamlit app for analyzing galvanostatic charge-discharge (GCD) curves.

Handles electrochemical energy storage devices. For a given discharge
curve it computes the gamma factor, the real and ideal energy, and the
corresponding power, then draws the energy region plot and a Ragone plot.
Results (figures and a summary table) can be exported as PDF.

Run with: streamlit run gcd_analyzer.py
"""

import io

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="GCD Analyzer", layout="centered")


# NumPy 2.0 renamed trapz to trapezoid. so it can Keep working on both old and new.
try:
    trapz = np.trapezoid
except AttributeError:
    trapz = np.trapz



# Print / PDF export of the interface
# 


_PRINT_CSS = """
<style>
@media print {
    /* Hide Streamlit chrome that shouldn't appear in the exported PDF. */
    header[data-testid="stHeader"],
    #MainMenu,
    footer,
    [data-testid="stToolbar"],
    .gcd-print-hide {
        display: none !important;
    }
    /* Give the printed page a little breathing room. */
    .block-container {
        padding-top: 1rem !important;
    }
}
</style>
"""


def enable_print_styles():
    """Inject the print stylesheet once per page load."""
    st.markdown(_PRINT_CSS, unsafe_allow_html=True)

def print_button(label):
    """A small button that opens the browser's print-to-PDF dialog.

    Rendered as an HTML component because it needs to call window.print()
    on the parent document, which plain Streamlit widgets can't do.
    """
    components.html(
        f"""
        <button onclick="window.parent.print()"
                style="padding:0.45rem 1rem; font-size:0.9rem;
                       border:1px solid #ccc; border-radius:0.4rem;
                       background:#f5f5f5; cursor:pointer;">
            {label}
        </button>
        """,
        height=48,
    )


# File loading

def load_data(uploaded_file):
    """Read an uploaded CSV or Excel file into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def clean_series(t, U):
    """Drop NaNs and make sure time is increasing.

    Real exported files sometimes carry a trailing blank row or a stray
    NaN, and occasionally the time column comes in reversed. We fix those
    quietly rather than blowing up mid-calculation.
    """
    t = np.asarray(t, dtype=float)
    U = np.asarray(U, dtype=float)

    good = ~(np.isnan(t) | np.isnan(U))
    t, U = t[good], U[good]

    # If time runs backwards, flip both arrays together.
    if len(t) > 1 and t[-1] < t[0]:
        t, U = t[::-1], U[::-1]

    return t, U



# Labels and units

def get_units(device_type, normalization_basis):
    """Return the display names and units for the chosen device/basis.

    Supercapacitors and mass-normalized batteries are per kg, a battery
    normalized by electrolyte volume is reported per liter (dm3).
    Three flavors of each unit are provided so it looks right everywhere:
      *_unit         -> plain ASCII, safe for CSV files
      *_unit_disp    -> Unicode superscript, for on-screen / PDF text
      *_unit_math    -> matplotlib mathtext, for plot axis labels
    """
    per_volume = (
        device_type == "Battery"
        and normalization_basis == "Electrolyte volume"
    )

    if per_volume:
        return {
            "energy_name": "Volumetric energy density",
            "power_name": "Volumetric power density",
            "energy_unit": "Wh dm-3",
            "power_unit": "W dm-3",
            "energy_unit_disp": "Wh dm\u207b\u00b3",
            "power_unit_disp": "W dm\u207b\u00b3",
            "energy_unit_math": r"$\mathrm{Wh\,dm^{-3}}$",
            "power_unit_math": r"$\mathrm{W\,dm^{-3}}$",
        }

    return {
        "energy_name": "Specific energy",
        "power_name": "Specific power",
        "energy_unit": "Wh kg-1",
        "power_unit": "W kg-1",
        "energy_unit_disp": "Wh kg\u207b\u00b9",
        "power_unit_disp": "W kg\u207b\u00b9",
        "energy_unit_math": r"$\mathrm{Wh\,kg^{-1}}$",
        "power_unit_math": r"$\mathrm{W\,kg^{-1}}$",
    }


# Core calculation
# ---------------------------------------------------------------------------

def calculate_metrics(t, U, current_A, device_type, normalization_basis,
                      active_mass_g=None, electrolyte_volume_dm3=None):
    """Compute gamma, energy and power from one discharge curve.
    """
    discharge_time = t[-1] - t[0]
    U_start = U[0]
    U_end = U[-1]

    # Measured area, trapezoidal rule.
    area_real = trapz(U, t)

    # Ideal reference area -- shape depends on the device.
    if device_type == "Supercapacitor":
        area_ideal = U_start * discharge_time / 2      # triangle
    else:
        area_ideal = U_start * discharge_time          # rectangle

    gamma = area_real / area_ideal

    # E = I * integral(U dt), still in joules at this point.
    energy_real_J = current_A * area_real
    energy_ideal_J = current_A * area_ideal

    # What we divide by: mass in kg, or volume already in dm3.
    if device_type == "Supercapacitor" or normalization_basis == "Active mass":
        norm_factor = active_mass_g / 1000.0           # g -> kg
    else:
        norm_factor = electrolyte_volume_dm3

    # Joules -> watt-hours (/3600), then normalize.
    energy_real = (energy_real_J / 3600) / norm_factor
    energy_ideal = (energy_ideal_J / 3600) / norm_factor
    energy_corrected = gamma * energy_ideal

    # Average power over the discharge, time expressed in hours.
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



# Plots

def build_energy_figure(t, U, device_type, discharge_time, U_start):
    """Measured curve plus the ideal and 'lost' energy regions.
    """
    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(t, U, color="black", linewidth=2, label="Experimental curve")

    # Area under the real curve = energy actually delivered.
    ax.fill_between(t, U, color="#cce5ff", alpha=0.8, label="Real energy")

    # Ideal curve on its own dense grid.
    t_ideal = np.linspace(0, discharge_time, 500)
    if device_type == "Supercapacitor":
        U_ideal = U_start * (1 - t_ideal / discharge_time)   # triangle
    else:
        U_ideal = np.full_like(t_ideal, U_start)             # rectangle

    ax.fill_between(t_ideal, U_ideal, color="#d4edda", alpha=0.5,
                    label="Ideal energy")

    # Gap between ideal and real: the part lost to non-ideality.
    U_on_ideal_grid = np.interp(t_ideal, t, U)
    ax.fill_between(t_ideal, U_on_ideal_grid, U_ideal,
                    where=(U_ideal > U_on_ideal_grid),
                    color="#ffe5cc", alpha=0.8, label="gamma loss")

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig


def build_ragone_figure(energy_value, power_value, energy_unit, power_unit,
                        device_name):
    """Single point on a log-log Ragone diagram (energy vs power)."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(energy_value, power_value, s=120, marker="o")
    ax.text(energy_value, power_value, device_name, fontsize=10)

    # Ragone plots are conventionally log-log.
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(f"Energy density ({energy_unit})")
    ax.set_ylabel(f"Power density ({power_unit})")
    ax.set_title("Ragone plot")
    ax.grid(which="both", alpha=0.3)
    fig.tight_layout()
    return fig


def build_table_figure(results, units):
    """Render the summary table as its own figure, so it can go to PDF.
    """
    # Short symbolic labels, matching the table used in the manuscript.
    rows = [
        ["\u03b3", "-", f"{results['gamma']:.4f}"],
        ["E_real", units["energy_unit_disp"], f"{results['energy_real']:.4f}"],
        ["E_ideal", units["energy_unit_disp"], f"{results['energy_ideal']:.4f}"],
        ["\u27e8E\u27e9", units["energy_unit_disp"],
         f"{results['energy_corrected']:.4f}"],
        ["P_real", units["power_unit_disp"], f"{results['power_real']:.4f}"],
    ]

    fig, ax = plt.subplots(figsize=(7, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Quantity", "Unit", "Value"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)
    fig.tight_layout()
    return fig


def figure_to_pdf_bytes(fig):
    """Serialize a matplotlib figure to PDF bytes for st.download_button."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()


def results_to_dataframe(results, units):
    """Flat table of the numbers, handy for the CSV download."""
    # The CSV is read on its own, away from the app, so spell things out
    # and keep units ASCII (Wh dm-3) so they survive any spreadsheet import.
    return pd.DataFrame(
        {
            "Quantity": [
                "gamma (deviation coefficient)",
                f"Real {units['energy_name'].lower()}",
                f"Ideal {units['energy_name'].lower()}",
                "Corrected energy (gamma x E_ideal)",
                f"Real {units['power_name'].lower()}",
            ],
            "Symbol": ["gamma", "E_real", "E_ideal", "<E>", "P_real"],
            "Unit": [
                "-",
                units["energy_unit"],
                units["energy_unit"],
                units["energy_unit"],
                units["power_unit"],
            ],
            "Value": [
                results["gamma"],
                results["energy_real"],
                results["energy_ideal"],
                results["energy_corrected"],
                results["power_real"],
            ],
        }
    )



# Results display

def display_results(results, units):
    """Numeric results as a readable block."""
    st.subheader("Results")
    st.write(f"\u03b3 = {results['gamma']:.4f}")
    st.write(f"Real {units['energy_name'].lower()}: "
             f"{results['energy_real']:.4f} {units['energy_unit_disp']}")
    st.write(f"Ideal {units['energy_name'].lower()}: "
             f"{results['energy_ideal']:.4f} {units['energy_unit_disp']}")
    st.write(f"Corrected energy (\u03b3 \u00d7 E_ideal): "
             f"{results['energy_corrected']:.4f} {units['energy_unit_disp']}")
    st.write(f"Real {units['power_name'].lower()}: "
             f"{results['power_real']:.4f} {units['power_unit_disp']}")


def generate_plots(t, U, results, units, device_type):
    """Draw both figures, show them, and offer PDF downloads."""
    energy_fig = build_energy_figure(
        t=t,
        U=U,
        device_type=device_type,
        discharge_time=results["discharge_time"],
        U_start=results["U_start"],
    )
    ragone_fig = build_ragone_figure(
        energy_value=results["energy_real"],
        power_value=results["power_real"],
        energy_unit=units["energy_unit_math"],
        power_unit=units["power_unit_math"],
        device_name=device_type,
    )
    table_fig = build_table_figure(results, units)

    st.subheader("Energy visualization")
    st.pyplot(energy_fig)
    st.download_button(
        "Download energy plot (PDF)",
        data=figure_to_pdf_bytes(energy_fig),
        file_name="energy_plot.pdf",
        mime="application/pdf",
    )

    st.subheader("Ragone plot")
    st.pyplot(ragone_fig)
    st.download_button(
        "Download Ragone plot (PDF)",
        data=figure_to_pdf_bytes(ragone_fig),
        file_name="ragone_plot.pdf",
        mime="application/pdf",
    )

    st.subheader("Summary table")
    st.pyplot(table_fig)

    # Two ways to grab the table: as a PDF (for the paper) or CSV (for reuse).
    col_pdf, col_csv = st.columns(2)
    with col_pdf:
        st.download_button(
            "Download table (PDF)",
            data=figure_to_pdf_bytes(table_fig),
            file_name="results_table.pdf",
            mime="application/pdf",
        )
    with col_csv:
        csv_bytes = results_to_dataframe(results, units).to_csv(
            index=False).encode("utf-8")
        st.download_button(
            "Download table (CSV)",
            data=csv_bytes,
            file_name="results_table.csv",
            mime="text/csv",
        )

# Input helpers


def collect_basic_inputs():
    """Current, device type and the two column names, in two columns."""
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
    """Ask for mass or electrolyte volume, depending on the device.

    Returns (basis, active_mass_g, electrolyte_volume_dm3); the value that
    doesn't apply stays None.
    """
    st.subheader("Normalization")

    normalization_basis = "Active mass"
    active_mass_g = None
    electrolyte_volume_dm3 = None

    if device_type == "Supercapacitor":
        active_mass_g = st.number_input("Active mass (g)",
                                        value=0.0004, format="%.6f")
        return normalization_basis, active_mass_g, electrolyte_volume_dm3

    # Battery: let the user choose how to normalize.
    normalization_basis = st.selectbox(
        "Normalization basis",
        ["Active mass", "Electrolyte volume"],
    )

    if normalization_basis == "Active mass":
        active_mass_g = st.number_input("Active mass (g)",
                                        value=0.0004, format="%.6f")
        return normalization_basis, active_mass_g, electrolyte_volume_dm3

    # Volume basis. Flow-battery electrolyte is usually dispensed per pole,
    # so we offer that and convert, but also accept a straight total volume.
    volume_unit = st.selectbox(
        "Electrolyte volume unit",
        ["dm3 (total)", "mL (each pole / electrode)"],
    )

    if volume_unit.startswith("mL"):
        volume_ml_per_pole = st.number_input(
            "Electrolyte volume per pole (mL)",
            value=800.0, format="%.2f")
        # per pole -> dm3, times 2 for anolyte + catholyte, since the
        # densities are reported on the total electrolyte volume.
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



# Main


def main():
    enable_print_styles()

    st.title("GCD-\u03b3 Analyzer")
    st.caption("Precise evaluation of energy and power characteristics "
               "in electrochemical energy storage devices")

    uploaded_file = st.file_uploader("Upload CSV or Excel file",
                                     type=["csv", "xlsx"])

    # Collect every setting first so the page layout doesn't jump around.
    current_A, device_type, time_col, voltage_col = collect_basic_inputs()
    norm_basis, active_mass_g, electrolyte_volume_dm3 = \
        collect_normalization_inputs(device_type)

    # Export of the interface as-is (before any data is loaded). Wrapped in a
    # container we can hide from the print itself so the button doesn't show.
    with st.container():
        st.markdown('<div class="gcd-print-hide">', unsafe_allow_html=True)
        print_button("Save interface as PDF")
        st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is None:
        st.info("Please upload a CSV or Excel file to begin.")
        return

    # Read the file and show a short preview.
    try:
        df = load_data(uploaded_file)
        st.subheader("Preview")
        st.dataframe(df.head())
    except Exception as err:
        st.error(f"Error reading file:\n{err}")
        return

    # Only crunch numbers once the user asks for it.
    if not st.button("Run analysis"):
        return

    # Grab the two columns we need.
    try:
        t = df[time_col].to_numpy()
        U = df[voltage_col].to_numpy()
    except KeyError:
        st.error("Column names not found. "
                 "Please verify the time and voltage columns.")
        return

    t, U = clean_series(t, U)
    if len(t) < 2:
        st.error("Not enough valid data points after cleaning.")
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

    # Export of the full page after analysis: preview + results + plots,
    # exactly as shown on screen.
    st.markdown('<div class="gcd-print-hide">', unsafe_allow_html=True)
    print_button("Save full results page as PDF")
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
