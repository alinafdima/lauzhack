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



@parser
def parseLidl(receipt):
    img, F, labels, ret = receipt.img, receipt.patches, receipt.conn_comp_labels, receipt.conn_comp_num
    
    i = 2
    while i < ret:
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
            i += 1
            textNext = F[i+1].getText()

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
            i += 1
            if pos4[1] < pos3[1]+10:
                i+= 1

                item["qty"] = F[i+1].getText()
                item["unitprice"] = F[i+2].getText()
            else:
                item["qty"] = F[i+1].getText()
                item["unitprice"] = ""
                if " " in item["qty"]:
                    item["qty"], item["unitprice"] = item["qty"].split(" ", 1)

        # showarray(subImg)
        # print item

        i+=1
        if levenshtein(item["title"], "zu zahlen") > 0.9:
            receipt.total = item["price"]
            break
        
        receipt.items.append(item)



@parser
def parseKarstadt(receipt):
    img, F, labels, ret = receipt.img, receipt.patches, receipt.conn_comp_labels, receipt.conn_comp_num
    pass
