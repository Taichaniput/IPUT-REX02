# core/financial_utils.py

def calculate_financial_indicators(financial_data):
    """財務指標を計算"""
    indicators = {}
    
    def safe_divide(a, b):
        if b and b != 0:
            return a / b
        return None
    
    indicators['roe'] = safe_divide(financial_data.net_income, financial_data.net_assets)
    indicators['roa'] = safe_divide(financial_data.net_income, financial_data.total_assets)
    indicators['operating_margin'] = safe_divide(financial_data.operating_income, financial_data.net_sales)
    indicators['asset_turnover'] = safe_divide(financial_data.net_sales, financial_data.total_assets)
    indicators['equity_ratio'] = safe_divide(financial_data.net_assets, financial_data.total_assets)
    
    return indicators