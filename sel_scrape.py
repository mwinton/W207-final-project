# Program to scrape insideschools.org
# We take DBNs from one of our CSVs.
# For each school, we check if:
# - it is a gifted school
# - it is a highly selective school
# We print out a CSV with the following columns:
# dbn,gifted,selective
# gifted can be: (0, 1)
# selective can be: (0, 1)

import urllib
from bs4 import BeautifulSoup
import csv
import time
import os

def get_dbns():
    dbns = []
    with open('data_cleaned/cleaned_shsat_outcomes.csv') as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            dbns.append(row['dbn'])
    return dbns

def get_school_info(dbn):
    info_page = 'https://insideschools.org/school/%s' % dbn
    print('requesting: %s' % info_page)
    page = urllib.request.urlopen(info_page)
    soup = BeautifulSoup(page, 'html.parser')
    with open("schoolinfo/%s.html" % dbn, "w") as file:
        file.write(str(soup))

def download_pages():
    dbns = get_dbns()
    for dbn in dbns:
        if os.path.exists('schoolinfo/%s.html' % dbn):
            print('skipping for: %s' % dbn)
            continue 
        print(dbn)
        get_school_info(dbn)
        time.sleep(3)

def print_school_category(override_list=None):
    schools = os.listdir(path='schoolinfo')
    if override_list is not None:
        schools = override_list
    print('dbn,gifted,selective')
    for school in schools:
        dbn = school.split('.')[0]
        school_path = 'schoolinfo/' + school
        school_fd = open(school_path, "r", encoding='utf-8')
        soup = BeautifulSoup(school_fd, "html.parser")
        divs = soup \
            .find_all("div", {"class": "school-icons"})
        for div in divs:
            spans = div.find_all("span")
            gifted = '0'
            selective = '0'
            for span in spans:
                #print(span)
                if "icon-gifted" in span['class']:
                    gifted = '1'
                if "icon-highly-selective" in span['class']:
                    selective = '1'
                if dbn == '18K235':    # Janice Marie Knight: SOAR
                    gifted = '1'
            print("%s,%s,%s" % (dbn, gifted, selective))

if __name__ == '__main__':
    download_pages()
    print_school_category()
