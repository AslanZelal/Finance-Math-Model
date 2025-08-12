# Finance-Math-Model
This model investigates the anomaly detection with machine learning using Hilger derivative and wavelet transform on the time scale of the sudden increase in the dollar exchange rate over the economic crisis that occurred in Türkiye around 2018.
import sys, re, time, os
import time

import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Used for scraping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import (
    TimeoutException,
    NoSuchElementException,
    ElementClickInterceptedException,
)



# Fetch all current S&P 500 tickers
def get_sp500_tickers():
    table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    df = table[0]
    tickers = df['Symbol'].tolist()
    return tickers

# Load the adjusted closing prices for a specific stock from Yahoo Finance
def load_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
    return stock_data['Adj Close']

# Example usage:

tickers = get_sp500_tickers()
#start_date = '1995-01-04'
#end_date = '2020-12-31'
start_date = '1999-01-04'
end_date = '2014-12-31'

data_frames = []

for ticker in tickers:
    try:
        data = load_data(ticker, start_date, end_date)
        data.name = ticker
        data_frames.append(data)
    except Exception as e:
        print(f"Failed to fetch data for {ticker} due to {e}")

# Concatenate all dataframes along axis 1
all_sp500_data_df = pd.concat(data_frames, axis=1)



def get_sse_380_companies_and_codes():
    base_url = "http://english.sse.com.cn/markets/indices/data/list/constituents/index.shtml?COMPANY_CODE=000009&INDEX_Code=000009"
    companies_and_codes = []

    # Initialize Selenium webdriver with WebDriver Manager
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # Run in headless mode (optional)
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--window-size=1920,1080')
    driver = webdriver.Chrome()

    try:
        # Navigate to the base URL
        driver.get(base_url)

        while True:
            try:
                # Wait for the table to load
                WebDriverWait(driver, 30).until(
                    EC.presence_of_element_located((By.XPATH, '//table[contains(@class, "js_constituentsList")]/tbody/tr'))
                )

                # Extract rows from the table
                rows = driver.find_elements(By.XPATH, '//table[contains(@class, "js_constituentsList")]/tbody/tr')
                for row in rows:
                    try:
                        name_cell = row.find_element(By.XPATH, './td[1]')
                        code_cell = row.find_element(By.XPATH, './td[2]')
                        
                        # Clean and format the stock code
                        stock_code = code_cell.text.strip()
                        if not stock_code.endswith(".SS"):
                            stock_code += ".SS"

                        companies_and_codes.append((name_cell.text.strip(), stock_code))
                    except Exception as e:
                        print(f"Error parsing row: {e}")

                # Attempt to find and click the "Next" button
                try:
                    next_button = WebDriverWait(driver, 10).until(
                        EC.element_to_be_clickable((By.XPATH, '//a[@class="flip" and @title="Next Page"]'))
                    )
                    # Scroll to the next button to ensure it's in view
                    driver.execute_script("arguments[0].scrollIntoView();", next_button)
                    time.sleep(1)  # Brief pause to ensure the element is in view
                    next_button.click()
                    print("Clicked on the Next button. Loading next page...")
                    # Wait for the new page's table to load
                    WebDriverWait(driver, 30).until(
                        EC.staleness_of(rows[0])  # Wait until the previous table is stale
                    )
                except TimeoutException:
                    print("No more pages found. Scraping complete.")
                    break  # Exit the loop if "Next" button is not found
                except ElementClickInterceptedException:
                    print("ElementClickInterceptedException encountered. Trying to click again.")
                    driver.execute_script("arguments[0].click();", next_button)
                    time.sleep(2)
                except NoSuchElementException:
                    print("No Next button found. Possibly reached the last page.")
                    break

            except TimeoutException:
                print("Timeout while waiting for the table to load. Taking a screenshot for debugging.")
                driver.save_screenshot('timeout_screenshot.png')
                print(driver.page_source)  # Optionally print the page source
                break  # Exit the loop or handle as needed

    finally:
        driver.quit()
    
    return companies_and_codes

companies_and_codes = get_sse_380_companies_and_codes()
print(companies_and_codes)

# Save to CSV
file_path = 'SSE-380_Tickers.csv'
df = pd.DataFrame(companies_and_codes, columns=['Company', 'Ticker'])
df.to_csv(file_path, index=False)





# Create a 3-row subplot
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 21))

# Plot data for each index on a separate subplot
all_sp500_data_df.iloc[:, :10].plot(ax=axes[0], title="Adjusted Closing Prices of S&P 500 Stocks (First 10)")
all_ftse250_data_df.iloc[:, :10].plot(ax=axes[1], title="Adjusted Closing Prices of FTSE 250 Stocks (First 10)")
all_sse380_data_df.iloc[:, :10].plot(ax=axes[2], title="Adjusted Closing Prices of SSE 380 Stocks (First 10)")

# Add common labels and grid
for ax in axes:
    ax.set_xlabel("Date")
    ax.set_ylabel("Adjusted Closing Price")
    ax.grid(True)

plt.tight_layout()
plt.show()

# Create a 3-row subplot
fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(15, 21))

# Function to plot column lengths in a subplot
def subplot_column_lengths(df, ax, title):
    column_lengths = df.count()
    sorted_column_lengths = column_lengths.sort_values(ascending=False)
    sorted_column_lengths.plot(kind='bar', ax=ax, color='blue', alpha=0.7)
    ax.set_title(title)
    ax.set_ylabel("Number of Data Points")
    ax.set_xlabel("Stock Ticker")

    # Optional: Display only a subset of tickers on the x-axis for clarity
    ax.set_xticks(list(range(0, len(sorted_column_lengths), int(len(sorted_column_lengths)/10))))
    ax.set_xticklabels(sorted_column_lengths.index[::int(len(sorted_column_lengths)/10)], rotation=45)
    
# Plotting for each index in subplots
subplot_column_lengths(all_sp500_data_df, axes[0], "Number of Data Points for Each S&P 500 Stock")
subplot_column_lengths(all_ftse250_data_df, axes[1], "Number of Data Points for Each FTSE 250 Stock")
subplot_column_lengths(all_sse380_data_df, axes[2], "Number of Data Points for Each SSE 380 Stock")

plt.tight_layout()
plt.show()


# Function to compute logarithmic returns
def compute_log_returns(df):
    # Take the logarithm of the adjusted closing prices
    log_prices = np.log(df)
    # Compute the returns: difference in log prices
    log_returns = log_prices.diff().iloc[1:]  # Skip the first NaN row resulting from diff()
    return log_returns

# Function to binarize the returns into +1 or -1
def binarize_returns(log_returns):
    s = log_returns.map(lambda x: 1 if x >= 0 else -1)
    return s

# Compute the mean and covariance matrix of the binary state sequences
def compute_C(s):
    T = s.shape[0]
    mean_s = s.mean(axis=0)
    E_sisj = (s.T @ s) / T
    C = E_sisj - np.outer(mean_s, mean_s)
    return C, mean_s

# Compute statistical moments of the coupling strengths
def compute_statistics(J_values):
    mean = np.mean(J_values)
    variance = np.var(J_values)
    skewness_value = skew(J_values)
    kurt = kurtosis(J_values, fisher=False)
    return mean, variance, skewness_value, kurt

# Shuffle the data to remove cross-correlations
def shuffle_data(s):
    s_shuffled = s.copy()
    for col in s_shuffled.columns:
        s_shuffled[col] = np.random.permutation(s_shuffled[col].values)
    return s_shuffled (!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!)




# Assuming filtered_sp500_data_df is your dataframe with stock data
correlation_matrix = filtered_sp500_data_df.corr()


# Extract the off-diagonal elements of the correlation matrix
off_diagonal_elements = correlation_matrix.values[np.triu_indices_from(correlation_matrix, k=1)]

mean_corr = np.mean(off_diagonal_elements)
median_corr = np.median(off_diagonal_elements)
mode_corr = pd.Series(off_diagonal_elements).mode()[0]  # Using Pandas to get the mode

print(f"Mean Correlation: {mean_corr}")
print(f"Median Correlation: {median_corr}")
print(f"Mode Correlation: {mode_corr}")

plt.figure(figsize=(15, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlation Matrix Heatmap")
plt.show()







# Compute the interaction strengths using the Naive Mean Field method
def compute_J_nMF(C, mean_s):
    A_diag = 1 - mean_s**2
    A_inv = np.diag(1 / A_diag)
    # Regularize the covariance matrix to ensure it's invertible
    epsilon = 1e-9
    C_reg = C + epsilon * np.eye(C.shape[0])
    C_inv = np.linalg.inv(C_reg)
    #C_inv = np.linalg.inv(C)
    J_nMF = A_inv - C_inv
    return J_nMF

# Compute the external fields using the Naive Mean Field method
def compute_h_nMF(J, mean_s):
    #epsilon = 1e-9
    epsilon = 0
    mean_s = np.clip(mean_s, -1 + epsilon, 1 - epsilon)
    h = np.arctanh(mean_s) - J @ mean_s
    return h

# Compute the interaction strengths using the TAP approximation
def compute_J_TAP(C_inv, mean_s):
    m_i = mean_s
    M = np.outer(m_i, m_i)
    discriminant = 1 - 8 * C_inv * M
    discriminant[discriminant < 0] = 0  # Handle small negative values due to numerical errors
    sqrt_term = np.sqrt(discriminant)
    denominator = 1 + sqrt_term
    #denominator[denominator == 0] = np.finfo(float).eps  # Avoid division by zero
    J_TAP = (-2 * C_inv) / denominator
    return J_TAP

# Compute the external fields using the TAP approximation
def compute_h_TAP(h_nMF, J_TAP, mean_s):
    m_i = mean_s
    one_minus_mj2 = 1 - mean_s**2
    J_TAP_squared = J_TAP**2
    correction = m_i * (J_TAP_squared @ one_minus_mj2)
    h_TAP = h_nMF - correction
    return h_TAP

# Extract the upper triangle values of the interaction matrix
def get_upper_triangle_values(J):
    upper_tri_indices = np.triu_indices_from(J, k=1)
    J_values = J[upper_tri_indices]
    return J_values
