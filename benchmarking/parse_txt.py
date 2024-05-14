import re

def parse(file_name):
  with open(file_name, "r") as file:
    with open("MSFT_benchmark.txt", "w") as writer:
      for line in file:
        if line.find("MainUnderlier") >= 0:
          security_price = parse_main_underlier(line)
        elif line.find("Expiration Symbol") >= 0:
          trading_time, r = parse_expiration(line)
        elif line.find("Strike") >= 0:
          vol, sp = parse_strike(line)
          writer.write(str(security_price) + " " + str(vol) + " " + str(sp) + " " + str(trading_time) + " " + str(r) + " \n")

def parse_main_underlier(line):
  bid = line.find("Bid")
  match = re.search(r'Bid="([^"]+)" Ask="([^"]+)"', line)
  
  try:
    bid_price = float(match.group(1))
    ask_price = float(match.group(2))
    return((bid_price + ask_price)/2)
  except Exception as e:
    print(e)
    print("Something went wrong in parse_main_underlier")
  

def parse_expiration(line):
  match_trading_time = re.search(r'TradingTime="([^"]+)"', line)
  match_sofr = re.search(r'SOFR="([^"]+)"', line)
  try:
    trading_time = float(match_trading_time.group(1))
    sofr = float(match_sofr.group(1))
    return trading_time, sofr
  except Exception as e:
    print(e)
    print("Something went wrong in parse_expiration")

def parse_strike(line):
  # Use regex to extract StrikeValue and ImplVol
  match_strike = re.search(r'StrikeValue="([^"]+)"', line)
  match_implvol = re.search(r'ImplVol="([^"]+)"', line)
  try:
    strike_value = float(match_strike.group(1))
    impl_vol = float(match_implvol.group(1))
    return impl_vol, strike_value
  except Exception as e:
    print(e)
    print("Something went wrong in parse_strike")


if __name__ == "__main__":
  file_names = ["MSFT.txt"]
  for fn in file_names:
    parse(fn)

