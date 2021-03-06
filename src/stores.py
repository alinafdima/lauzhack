from utils import *
import re
import ipdb


Classes = {}
def parser(type = None, name = None):
    def _parser(type):
        _name = name
        if _name is None:
            _name = type.__name__[5:]
        Classes[_name] = type
        return type

    if type is None:
        return _parser
    else:
        return _parser(type)


def parse_date(receipt):
    regex_pattern = re.compile('[0-9]{1,2}[\\./,\\-][0-9]{1,2}[\\./,\\-][0-9]{2,4}')

    m = regex_pattern.search(receipt.img_text)
    if m:
        receipt.date = m.group(0)
        return

    for patch in receipt.patches:
        m = regex_pattern.search(patch.getText())
        if m:
            # print m.group(0), patch.getText()
            receipt.date = m.group(0)
            return


def parse_total(receipt):
    regex_pattern = re.compile('^(zuzahlen|betrag|gesamt|summe):?(eur)?([0-9]+[\\.,][0-9]+)')
    processed_text = receipt.img_text.lower().replace(' ', '')
    m = regex_pattern.search(processed_text)
    if m:
        total_raw = m.group(3)
        total_raw = total_raw.replace(',', '.')
        receipt.total = float(total_raw)
    else:
        regex_pattern2 = re.compile('^(total):?(eur)?([0-9]+[\\.,][0-9]+)')
        m = regex_pattern2.search(processed_text)

        if m:
            total_raw = m.group(3)
            total_raw = total_raw.replace(',', '.')
            receipt.total = float(total_raw)

@parser
def parseLidl(receipt):
    F = receipt.patches
    
    i = 1
    while i < len(F):
        subImg, pos = F[i].img, F[i].bbox
        text = F[i].getText()

        # print "text", text
        if not text:
            i += 1
            continue

        if levenshtein(text, "Halbergstr. 3") > 0.9 or levenshtein(text, "66121 Saarbricken") > 0.9 or \
                levenshtein(text, "Mo-Sa 8-20 Uhr /") > 0.9 or levenshtein(text, "So geschlossen") > 0.9 or \
                levenshtein(text, "EUR") > 0.9:
            i += 1
            continue

        item = {}

        subImgNext, posNext = F[i+1].img, F[i+1].bbox
        if posNext[1] < pos[1]+10:
            textNext = F[i+1].getText()
            i += 1

            item["title"] = text
            item["price"] = textNext
            item["vat"] = ""
            if " " in textNext:
                item["price"], item["vat"] = textNext.split(" ", 1)
        else:
            A = text.split(" ", 2)
            item["title"] = A[0]
            item["price"] = item["vat"] = ""
            if len(A) > 1:
                item["price"] = A[1]
            if len(A) > 2:
                item["vat"] = A[2]

        subImg3, pos3 = F[i+1].img, F[i+1].bbox
        subImg4, pos4 = F[i+2].img, F[i+2].bbox

        if pos3[0] > pos[0] + 20:
            text3 = F[i+1].getText()
            text4 = F[i+2].getText()
            i += 1

            if pos4[1] < pos3[1]+10:
                i+= 1

                item["qty"] = text3
                item["unitprice"] = text4
            else:
                item["qty"] = text3
                item["unitprice"] = ""
                if " " in item["qty"]:
                    item["qty"], item["unitprice"] = item["qty"].split(" ", 1)
            item["qty"] = ''.join(c for c in item["qty"] if c.isdigit())

        # showarray(subImg)
        # print item

        # qty_str = ""
        # if "qty" in item:
        #     qty_str = "%s x %s"%(item["qty"], item["unitprice"])
        # print "%50s %10s %s, VAT %s"%(item["title"], qty_str, item["price"], item["vat"])

        i+=1
        if levenshtein(item["title"], "zu zahlen") > 0.9:
            receipt.total = item["price"]
            break
        
        receipt.items.append(item)



@parser
def parseKarstadt(receipt):
    F = receipt.patches
    i = 1

    runningTotal = 0
    while i < len(F):
        subImg, pos = F[i].img, F[i].bbox
        text = F[i].getText()

        if not text:
            i += 1
            continue

        item = {}

        subImgNext, posNext = F[i+1].img, F[i+1].bbox
        if posNext[1] < pos[1]+10:
            textNext = F[i+1].getText()
            i += 1

            item["title"] = text
            item["price"] = textNext
            item["vat"] = ""
            if " " in textNext:
                item["price"], item["vat"] = textNext.split(" ", 1)
        else:
            i += 1
            continue

            A = text.split(" ", 2)
            item["title"] = A[0]
            item["price"] = item["vat"] = ""
            if len(A) > 1:
                item["price"] = A[1]
            if len(A) > 2:
                item["vat"] = A[2]

        # print item

        if item["price"] and not is_number(item["price"]):
            break

        if item["price"]:
            runningTotal += to_number(item["price"], dft = 0)

        i+=1
        keywords = ["zu zahlen", "betrag", "gesamt", "summe", "total", "eur"]
        score = max( levenshtein(item["title"].lower(), word) for word in keywords )
        if score > 0.9:
            total = item["price"] + item["vat"]
            total = to_number(total, dft = 0)
            if total > 0:
                receipt.total = total

            break
        
        receipt.items.append(item)

    if not receipt.total and runningTotal > 0:
        receipt.total = str(runningTotal)
