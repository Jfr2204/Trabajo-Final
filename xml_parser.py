import xml.etree.ElementTree as ET
import pandas as pd

data = []

FILE_PATH = "data/en_product6.xml"

tree = ET.parse(FILE_PATH)
root = tree.getroot()

disorders = root.findall("DisorderList/Disorder")

for disorder in disorders:
    name = disorder.find("Name").text
    orphacode = disorder.find("OrphaCode").text
    
    genes = disorder.findall(".//Gene")
    if len(genes) == 0:
        data.append({
            "OrphaCode": orphacode,
            "DisorderName": name,
            "GeneSymbol": None,
            "GeneName": None,
            "GeneLocus": None
        })
    else:
        for gene in genes:
            symbol = gene.find("Symbol").text if gene.find("Symbol") is not None else None
            fullname = gene.find("Name").text if gene.find("Name") is not None else None
            
            loci = []
            locus_list = gene.find("LocusList")
            if locus_list is not None:
                for locus in locus_list.findall("Locus"):
                    gene_locus = locus.find("GeneLocus").text if locus.find("GeneLocus") is not None else None
                    loci.append(gene_locus)

            gene_locus_str = ",".join(loci) if loci else None

            data.append({
                "OrphaCode": orphacode,
                "DisorderName": name,
                "GeneSymbol": symbol,
                "GeneName": fullname,
                "GeneLocus": gene_locus_str 
            })

df = pd.DataFrame(data)
df.to_csv("data/disorder_genes.csv", index=False)
