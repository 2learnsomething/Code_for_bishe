import requests
import openpyxl as oxl
import time

def download(code, start_date, end_date):
    download_url = (
        "http://quotes.money.163.com/service/chddata.html?code=1"
        + code
        + "&start="
        + start_date
        + "&end="
        + end_date
        + "&fields=TCLOSE;HIGH;LOW;TOPEN;LCLOSE;CHG;PCHG;TURNOVER;VOTURNOVER;VATURNOVER;TCAP;MCAP"
    )
    print(download_url)
    data = requests.get(download_url, headers=headers)
    f = open('stock_price/'+code + ".csv", "wb")
    for chunk in data.iter_content(chunk_size=10000):
        if chunk:
            f.write(chunk)
    print("股票---", code, "历史数据正在下载")

def get_stock_code(file_path):
    code_list = []
    sheets = oxl.load_workbook(file_path).active
    for i in range(2,sheets.max_row+1):
        code_list.append(sheets['E'+str(i)].value)
    return code_list
    
if __name__ == '__main__':
    headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36"
}
    start_date = '20160601'
    end_date = '20211231'
    file_path = r'深交所A股列表_主板.xlsx'
    code_list = get_stock_code(file_path)
    print("--------开始抓取---------")
    for code in code_list:
        download(code, start_date, end_date)
        time.sleep(2)
    print("--------抓取完成---------")
