import sys, re, csv
from Bio import SeqIO

input_file_name = sys.argv[1]
output_file_name = sys.argv[2]


with open(input_file_name, mode='r') as handle:
    output_list = []
    for record in SeqIO.parse(handle, 'fasta'):
        output_list.append([re.sub("sp\||\|.*", "", record.id), str(record.seq)])

with open(output_file_name, "w", newline="") as file:
    writer = csv.writer(file)
    for line in output_list:
        writer.writerow(line)
