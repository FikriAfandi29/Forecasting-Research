"""
==================================================================
BENCHMARK FORECASTING XAU/USD (HARIAN): LSTM vs Gemini 2.5 Pro
==================================================================
Eksperimen hybrid & benchmarking untuk riset komputasi ekonomi:
1. LSTM dilatih pada 80% data historis numerik (time series harian).
2. Gemini 2.5 Pro diberi data numerik dalam bentuk teks
   (context-window prompting) untuk memprediksi 20% sisa data.
3. Kedua hasil dibandingkan dengan metrik: MSE, MAE, RMSE, MAPE.

Koneksi ke Gemini menggunakan Vertex AI (project-based), bukan API key —
jadi butuh project GCP yang sudah diaktifkan Vertex AI API-nya.

Cara pakai di Google Colab:
    1. Jalankan cell instalasi (paling atas).
    2. Autentikasi GCP dulu (lihat catatan di bagian CONFIG).
    3. Isi PROJECT_ID & LOCATION di bagian CONFIG.
    4. Pastikan file CSV sudah di-upload dengan kolom:
        Date, XAU_USD, Oil_Price, US_CPI, Fed_Rate, Global_Commodity_Index
    5. Run all.

CATATAN PERBAIKAN DARI VERSI SEBELUMNYA:
- thinking_budget TIDAK lagi -1 (dynamic/unlimited). Di Gemini 2.5 Pro,
  token "thinking" ikut dipotong dari total max_output_tokens, sehingga
  jika thinking terlalu besar, output JSON bisa TERPOTONG di tengah
  (menyebabkan error "Expecting ',' delimiter" saat json.loads()).
  Sekarang thinking_budget dibatasi eksplisit (2048) supaya output JSON
  selalu punya ruang token yang cukup.
- max_output_tokens dinaikkan ke 32768 agar aman untuk ratusan baris
  hasil prediksi harian + sisa token thinking.
- Ditambahkan retry sekali dengan token lebih besar jika percobaan
  pertama gagal parse, sebelum jatuh ke fallback baseline naif.
- Ditambahkan logging panjang raw response saat gagal, untuk debugging.
==================================================================
"""

# ==========================================
# 0. RUN INSTALASI & AUTENTIKASI (JIKA BELUM)
# ==========================================
# !pip install -q google-genai tensorflow scikit-learn matplotlib pandas yfinance requests
# from google.colab import auth
# auth.authenticate_user()

import json
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

from google import genai
from google.genai import types

from google.colab import auth
auth.authenticate_user()


# ==========================================
# CONFIG (DAILY)
# ==========================================
CONFIG = {
    "csv_file": "XAU_Macro_Dataset_Fixed.csv",  # Dataset harian bebas NaN

    # --- Vertex AI (project-based, tanpa API key) ---
    "project_id": "gen-lang-client-0971794485",
    "location": "us-central1",
    "gemini_model": "gemini-2.5-pro",

    # --- Parameter generasi Gemini 2.5 Pro ---
    "thinking_budget": 2048,      # dibatasi eksplisit, BUKAN -1, agar output tidak terpotong
    "max_output_tokens": 32768,   # ruang aman untuk thinking + JSON ratusan angka
    "temperature": 0.0,           # deterministik, ideal untuk prediksi angka keuangan

    "features": ["XAU_USD", "Oil_Price", "US_CPI", "Fed_Rate", "Global_Commodity_Index"],
    "target_col": "XAU_USD",
    "train_ratio": 0.80,
    "seq_length": 30,       # jumlah hari historis yang dilihat LSTM tiap prediksi
    "sample_stride": 5,     # ambil 1 baris tiap 5 hari untuk konteks teks Gemini (hemat token)
    "gemini_batch_size": 60,  # pecah test period jadi batch kecil agar respons tidak terpotong
    "lstm_epochs": 15,
    "lstm_batch_size": 32,
}


# ==========================================
# 1. DATA LOADING & SPLIT
# ==========================================
def load_and_split_data(csv_file: str, train_ratio: float):
    df = pd.read_csv(csv_file)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    split_idx = int(len(df) * train_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"📊 Total data: {len(df)} hari")
    print(
        f"📈 Training ({train_ratio:.0%}): {len(train_df)} hari "
        f"({train_df['Date'].min():%Y-%m-%d} s.d {train_df['Date'].max():%Y-%m-%d})"
    )
    print(
        f"📉 Testing ({1 - train_ratio:.0%}): {len(test_df)} hari "
        f"({test_df['Date'].min():%Y-%m-%d} s.d {test_df['Date'].max():%Y-%m-%d})"
    )
    return train_df, test_df


# ==========================================
# 2. BASELINE MODEL: LSTM
# ==========================================
def create_sequences(data: np.ndarray, seq_length: int):
    """Ubah data 2D (timesteps x fitur) jadi sequence untuk LSTM."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length, :])
        y.append(data[i + seq_length, 0])  # target = kolom pertama (XAU_USD)
    return np.array(X), np.array(y)


def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def run_lstm_forecast(train_df, test_df, features, seq_length, epochs, batch_size):
    print("\n🤖 [1/2] Melatih model LSTM...")

    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(train_df[features])
    scaled_test = scaler.transform(test_df[features])

    X_train, y_train = create_sequences(scaled_train, seq_length)

    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    # Gabungkan ekor training + seluruh test agar sequence pertama test tetap valid
    combined = np.vstack((scaled_train[-seq_length:], scaled_test))
    X_test, _ = create_sequences(combined, seq_length)
    predictions_scaled = model.predict(X_test, verbose=0)

    # Inverse transform ke skala harga asli
    dummy = np.zeros((len(predictions_scaled), len(features)))
    dummy[:, 0] = predictions_scaled.flatten()
    predictions = scaler.inverse_transform(dummy)[:, 0]

    print("✅ Forecasting LSTM selesai.")
    return predictions


# ==========================================
# 3. PROPOSED MODEL: GEMINI 2.5 PRO
# ==========================================
def build_history_text(train_df: pd.DataFrame, stride: int) -> str:
    """Sampling data training tiap `stride` hari agar hemat token."""
    lines = []
    for _, row in train_df.iloc[::stride].iterrows():
        lines.append(
            f"Date:{row['Date']:%Y-%m-%d}, XAU:{row['XAU_USD']:.2f}, "
            f"Oil:{row['Oil_Price']:.2f}, CPI:{row['US_CPI']:.2f}, "
            f"Fed:{row['Fed_Rate']:.2f}, CommIndex:{row['Global_Commodity_Index']:.2f}"
        )
    return "\n".join(lines)


def build_prediction_prompt(history_text: str, test_df: pd.DataFrame) -> str:
    factor_lines = [
        f"Date: {row['Date']:%Y-%m-%d} -> Oil:{row['Oil_Price']:.2f}, "
        f"CPI:{row['US_CPI']:.2f}, Fed:{row['Fed_Rate']:.2f}, "
        f"CommIndex:{row['Global_Commodity_Index']:.2f}"
        for _, row in test_df.iterrows()
    ]

    return f"""You are an expert quantitative financial economist. I will provide a historical
training dataset of XAU/USD (Gold) prices along with macroeconomic factors, then the macro
factors for the test period. Predict the XAU_USD price for each date in the test period.

[HISTORICAL TRAINING DATA TREND]
{history_text}

[TEST PERIOD FACTOR INPUTS]
{chr(10).join(factor_lines)}

Respond ONLY with a valid JSON array of numbers representing the predicted XAU_USD prices,
in the exact chronological order of the test period dates. No text, no markdown, no explanation.

STRICT NUMBER FORMAT RULES (very important):
- Do NOT use thousand separators (no commas inside numbers). Write 3245.67, NOT 3,245.67.
- Use plain decimal numbers only, with a dot as the decimal separator.
- Do NOT add trailing commas after the last number in the array.
- Output must be a single line, valid, parseable JSON array with no line breaks between numbers.

Example: [1950.25, 1962.40, 1945.10, ...]
"""


def _extract_numbers_via_regex(text: str):
    """
    Fallback parser: ekstrak semua angka desimal dari teks tanpa bergantung
    pada validitas struktur JSON. Berguna saat Gemini menulis koma ribuan,
    trailing comma, atau newline yang merusak json.loads().

    Membuang koma yang menempel di antara digit (contoh: 3,245.67 -> 3245.67)
    sebelum mengambil semua pola angka desimal.
    """
    # Hilangkan koma yang berfungsi sebagai pemisah ribuan (digit,digit)
    cleaned = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", text)
    # Ambil semua angka (boleh negatif, boleh desimal)
    matches = re.findall(r"-?\d+\.?\d*", cleaned)
    return [float(m) for m in matches if m not in ("", "-", ".")]


def _call_gemini_and_parse(client, model_name, prompt, thinking_budget,
                            max_output_tokens, temperature):
    """Satu kali panggilan ke Gemini + parsing hasil. Melempar exception jika gagal total."""
    gen_config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget),
        max_output_tokens=max_output_tokens,
        temperature=temperature,
    )

    response = client.models.generate_content(
        model=model_name, contents=prompt, config=gen_config
    )

    raw_text = response.text or ""
    clean_text = raw_text.replace("```json", "").replace("```", "").strip()

    if not clean_text:
        raise ValueError(
            f"Respons kosong (kemungkinan seluruh token habis untuk 'thinking'). "
            f"finish_reason: {getattr(response.candidates[0], 'finish_reason', 'unknown') if response.candidates else 'unknown'}"
        )

    # --- Coba parse JSON standar dulu ---
    try:
        return json.loads(clean_text)
    except json.JSONDecodeError as e:
        print(f"⚠️ json.loads() gagal ({e}). Mencoba fallback ekstraksi angka via regex...")
        numbers = _extract_numbers_via_regex(clean_text)
        if not numbers:
            raise ValueError("Fallback regex juga tidak menemukan angka apa pun di respons.")
        print(f"✅ Fallback regex berhasil mengekstrak {len(numbers)} angka.")
        return numbers


def _forecast_one_batch(client, model_name, history_text, batch_test_df,
                         fallback_value, thinking_budget, max_output_tokens, temperature):
    """Jalankan satu batch prediksi (percobaan pertama + retry + fallback)."""
    prompt = build_prediction_prompt(history_text, batch_test_df)
    n_expected = len(batch_test_df)
    predictions = None

    try:
        predictions = _call_gemini_and_parse(
            client, model_name, prompt, thinking_budget, max_output_tokens, temperature
        )
    except Exception as e:
        print(f"   ⚠️ Batch gagal percobaan pertama ({e}). Retry dengan token lebih besar...")
        try:
            predictions = _call_gemini_and_parse(
                client, model_name, prompt,
                thinking_budget=512,
                max_output_tokens=max_output_tokens * 2,
                temperature=temperature,
            )
            print("   ✅ Retry batch berhasil.")
        except Exception as e2:
            print(f"   ❌ Retry batch juga gagal ({e2}). Menggunakan baseline naif untuk batch ini.")
            predictions = [fallback_value] * n_expected

    if len(predictions) != n_expected:
        print(
            f"   ⚠️ Penyesuaian panjang batch: dapat {len(predictions)}, "
            f"ekspektasi {n_expected}."
        )
        predictions = (list(predictions) + [fallback_value] * n_expected)[:n_expected]

    return predictions


def run_gemini_forecast(client, model_name, train_df, test_df, sample_stride,
                        fallback_value, thinking_budget, max_output_tokens, temperature,
                        batch_size):
    print("\n🧠 [2/2] Menjalankan forecasting Gemini 2.5 Pro (mode batch)...")

    history_text = build_history_text(train_df, sample_stride)
    n_total = len(test_df)
    n_batches = int(np.ceil(n_total / batch_size))
    all_predictions = []

    # fallback value bergerak: kalau satu batch gagal total, pakai nilai aktual
    # terakhir yang diketahui sebelum batch itu (lebih masuk akal daripada
    # selalu memakai harga terakhir training untuk batch yang jauh di masa depan)
    rolling_fallback = fallback_value

    for i in range(n_batches):
        start = i * batch_size
        end = min(start + batch_size, n_total)
        batch_test_df = test_df.iloc[start:end]

        print(f"\n   ▶️ Batch {i + 1}/{n_batches} ({batch_test_df['Date'].min():%Y-%m-%d} "
              f"s.d {batch_test_df['Date'].max():%Y-%m-%d}, {len(batch_test_df)} hari)")

        batch_predictions = _forecast_one_batch(
            client, model_name, history_text, batch_test_df,
            fallback_value=rolling_fallback,
            thinking_budget=thinking_budget,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
        )
        all_predictions.extend(batch_predictions)

        # update rolling fallback pakai prediksi terakhir batch ini (untuk batch berikutnya)
        if len(batch_predictions) > 0:
            rolling_fallback = batch_predictions[-1]

    print("\n✅ Forecasting Gemini 2.5 Pro selesai (semua batch).")
    return np.array(all_predictions[:n_total], dtype=float)


# ==========================================
# 4. EVALUASI METRIK
# ==========================================
def calculate_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(actual, predicted) * 100
    return mse, mae, rmse, mape


def print_metrics_table(y_true, lstm_pred, gemini_pred):
    mse_l, mae_l, rmse_l, mape_l = calculate_metrics(y_true, lstm_pred)
    mse_g, mae_g, rmse_g, mape_g = calculate_metrics(y_true, gemini_pred)

    metrics_df = pd.DataFrame({
        "Metrik Evaluasi": ["MSE", "MAE", "RMSE", "MAPE"],
        "LSTM Model": [f"{mse_l:.4f}", f"{mae_l:.2f}", f"{rmse_l:.2f}", f"{mape_l:.2f}%"],
        "Gemini 2.5 Pro": [f"{mse_g:.4f}", f"{mae_g:.2f}", f"{rmse_g:.2f}", f"{mape_g:.2f}%"],
    })

    print("\n📊 ================= TABEL PERBANDINGAN PERFORMA =================")
    print(metrics_df.to_string(index=False))
    return metrics_df


# ==========================================
# 5. VISUALISASI
# ==========================================
def plot_forecast_comparison(test_df, y_true, lstm_pred, gemini_pred):
    plt.figure(figsize=(14, 7))
    plt.plot(test_df["Date"], y_true, label="Actual XAU_USD", color="black", linewidth=2)
    plt.plot(test_df["Date"], lstm_pred, label="LSTM Forecast", color="blue", linestyle="--")
    plt.plot(test_df["Date"], gemini_pred, label="Gemini 2.5 Pro Forecast", color="orange", linestyle="-.")
    plt.title("XAU/USD Monthly Forecast Comparison (20% Test Data)")
    plt.xlabel("Date")
    plt.ylabel("Gold Price (USD)")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()
    plt.show()


# ==========================================
# MAIN
# ==========================================
def main(config: dict):
    train_df, test_df = load_and_split_data(config["csv_file"], config["train_ratio"])
    y_true = test_df[config["target_col"]].values

    lstm_predictions = run_lstm_forecast(
        train_df, test_df,
        features=config["features"],
        seq_length=config["seq_length"],
        epochs=config["lstm_epochs"],
        batch_size=config["lstm_batch_size"],
    )

    # Inisialisasi client via Vertex AI (project-based)
    client = genai.Client(
        vertexai=True,
        project=config["project_id"],
        location=config["location"],
    )
    gemini_predictions = run_gemini_forecast(
        client, config["gemini_model"],
        train_df, test_df,
        sample_stride=config["sample_stride"],
        fallback_value=train_df[config["target_col"]].iloc[-1],
        thinking_budget=config["thinking_budget"],
        max_output_tokens=config["max_output_tokens"],
        temperature=config["temperature"],
        batch_size=config["gemini_batch_size"],
    )

    print_metrics_table(y_true, lstm_predictions, gemini_predictions)
    plot_forecast_comparison(test_df, y_true, lstm_predictions, gemini_predictions)


if __name__ == "__main__":
    main(CONFIG)
