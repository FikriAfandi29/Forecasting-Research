import pandas as pd
import yfinance as yf
import requests

# Konfigurasi
FRED_API_KEY = "" 
START_DATE = "2015-01-01"
END_DATE = "2025-12-30"

print("🔄 [1/3] Menarik data harian dari Yahoo Finance...")
tickers = {"XAU_USD": "GC=F", "Oil_Price": "CL=F"}
df_yf = pd.DataFrame()

for name, ticker in tickers.items():
    data = yf.download(ticker, start=START_DATE, end=END_DATE, auto_adjust=True)
    if not data.empty:
        if isinstance(data.columns, pd.MultiIndex):
            df_yf[name] = data.iloc[:, data.columns.get_level_values(0) == 'Close'].iloc[:, 0]
        else:
            df_yf[name] = data['Close']

df_yf = df_yf.reset_index()
df_yf['Date'] = pd.to_datetime(df_yf['Date']).dt.tz_localize(None)

# Fungsi Ambil FRED murni tanpa paksa convert pembulatan int
def fetch_fred_exact(series_id, col_name, api_key):
    url = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&api_key={api_key}&file_type=json"
    response = requests.get(url)
    if response.status_code == 200:
        obs = response.json().get('observations', [])
        if obs:
            df = pd.DataFrame(obs)[['date', 'value']]
            df['date'] = pd.to_datetime(df['date'])
            # Mengonversi ke float secara presisi demi mempertahankan desimal panjang
            df['value'] = pd.to_numeric(df['value'], errors='coerce')
            print(f"✅ Berhasil menarik {series_id} dengan desimal presisi.")
            return df.rename(columns={'date': 'Date', 'value': col_name})
    print(f"❌ Gagal mengambil series {series_id}")
    return pd.DataFrame()

print("\n🔄 [2/3] Menarik data makro & Indeks Komoditas Global Desimal...")
df_cpi = fetch_fred_exact("CPIAUCSL", "US_CPI", FRED_API_KEY)
df_fed = fetch_fred_exact("FEDFUNDS", "Fed_Rate", FRED_API_KEY)
# PALLFNFINDEXM adalah index bulanan IMF yang memuat desimal panjang persis seperti contohmu
df_comm = fetch_fred_exact("PALLFNFINDEXM", "Global_Commodity_Index", FRED_API_KEY)

# ==========================================
# PROSES MERGE AMAN & PROPAGASI DESIMAL
# ==========================================
print("\n🔄 [3/3] Sinkronisasi rentang waktu dan ekspor CSV...")
date_range = pd.date_range(start=START_DATE, end=END_DATE)
df_final = pd.DataFrame({'Date': date_range})

df_final = pd.merge(df_final, df_yf, on='Date', how='left')
if not df_cpi.empty: df_final = pd.merge(df_final, df_cpi, on='Date', how='left')
if not df_fed.empty: df_final = pd.merge(df_final, df_fed, on='Date', how='left')
if not df_comm.empty: df_final = pd.merge(df_final, df_comm, on='Date', how='left')

# Forward fill harian untuk mempertahankan nilai desimal bulanan sepanjang hari aktif
df_final = df_final.ffill().bfill()

output_file = "XAU_Macro_Dataset_Fixed.csv"
df_final.to_csv(output_file, index=False)

print(f"\n🚀 Dataset Sukses Dibuat!")
print(f"📂 File disimpan di panel Colab dengan nama: {output_file}")
print("\nPengecekan 5 baris data teratas:")
print(df_final[['Date', 'XAU_USD', 'Global_Commodity_Index']].head(5).to_string(index=False))
