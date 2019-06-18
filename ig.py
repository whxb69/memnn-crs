from math import log
import re
import operator

def data():
    f = open(r'personalized-dialog-dataset\full\personalized-dialog-task3-options-trn.txt','r')
    text = f.read()
    dias = text.split('\n\n')
    print(len(dias))
    pairs = []
    pwords = {}
    rwords = {}
    for dia in dias[:-1]:
        reg = re.search('resto.+_\d\n',dia).regs[0]
        res = dia[reg[0]:reg[1]-1]
        ts = dia.split('\n')

        profile = ts[0]
        profile = re.sub('\d+ ', '', profile)
        for word in profile.split(' '):
            if word not in pwords.keys():
                pwords[word] = 0
            else:
                pwords[word] = pwords[word] + 1
        if res not in rwords.keys():
            rwords[res] = 0
        else:
            rwords[res] = rwords[res] + 1
        pairs.append([profile,res])

    jpd = {}

    lenp = len(pairs)
    for key in pwords.keys():
        pwords[key] = round(pwords[key] / lenp,3)
        for pair in pairs:
            if key in pair[0] and key + '\t' + pair[1] not in jpd.keys():
                num = 0
                res = pair[1]
                sign = key + '\t' + res
                for pair in pairs:
                    if key in pair[0] and pair[1] == res:
                        num += 1
                jpd[sign] = round(num/lenp,5)

    for key in rwords.keys():
        rwords[key] = round(rwords[key] / lenp,5)

    print(pwords)
    print(rwords)
    print(jpd)
    return pwords,rwords,pairs,jpd

def Gain(i, e):
    return i - e

def main():
    gains = {}
    pwords,rwords,paris,jpd = data()
    for key in pwords.keys():
        for pair in paris:
            if key in pair[0] and key + '\t' + pair[1] not in gains.keys():
                res = pair[1]
                pr = rwords[res]
                pp = pwords[key]
                hx = pr * log(pr, 2)
                hx = -hx
                sign = key + '\t' + res
                hxz = jpd[sign] * log(pp/jpd[sign],2)

                gain = hx - hxz
                gains[sign] = round(gain,5)
    return gains


if __name__ == "__main__":
    gains = main()
    res = sorted(gains.items(),key=operator.itemgetter(1),reverse=True)
    for r in res:
        print(r)
