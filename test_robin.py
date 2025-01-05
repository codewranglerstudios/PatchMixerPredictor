import robin_stocks.robinhood as r

# Login to Robinhood (Add 2FA handling if needed)
login = r.authentication.login("hello@codewranglerstudios.com", "07224343dbjfj7fH!")

# Get a stock quote
quote = r.stocks.get_quotes("AAPL")
print(quote)

# Place a buy order (uncomment with caution)
# r.orders.order_buy_market("AAPL", 1)

# Place a sell order (uncomment with caution)
# r.orders.order_sell_market("AAPL", 1)

# Get your portfolio
portfolio = r.profiles.load_portfolio_profile()
print(portfolio)

# Logout
r.authentication.logout()
