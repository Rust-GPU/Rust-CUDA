# Takes the pdf file of the libcuda docs and generates a JSON file representing it.
# That json file is then used to generate internal intrinsics as well as intrinsics docs.
# libdevice is 300+ intrinsics, therefore making a script to do this is better for developer
# sanity as well as extensibility for any future versions of libdevice.

import pdfplumber
import os
import re
import json

dirname = os.path.dirname(__file__)
filename = os.path.join(dirname, 'data/libdevice.pdf')
out_filename = os.path.join(dirname, 'data/libdevice.json')

text = ""
with pdfplumber.open(filename) as pdf:
    for page in pdf.pages:
        text += page.extract_text()

open("scripts/data/libdevice.txt", "w", encoding="utf8").write(text)

# I know this is bad but trust me its much less work than writing a proper parser
regex = r"3\.\d+\.\s(\w+)(?!\.)\nPrototype:\n(.+)\nDescription:\n([\s\S]*?(?=Returns:))Returns:\n([\s\S]*?(?=Library Availability))Library Availability:\n([\s\S]*?(?=(3\.\d+\.)|\Z|www\.nvidia\.com))"

# The raw text includes the page footer which messes up the regex so clean that up before we go on
sanitize_regex = r"www.nvidia.com\nLibdevice User's Guide Part 000 _v8.0 \| \d+Function Reference\n"

text = re.sub(sanitize_regex, "", text)
# renders better in markdown
text = text.replace("\u2023", "-")
# replace more than one space in a row with a single space
text = re.sub(" +", " ", text)
# the text conversion has some issues with the math symbols in the pdf
# it seems to turn x and y into \nx and \ny
text = text.replace("\nx", "x")
text = text.replace("\ny", "y")
# i dont even know
text = text.replace(".x", "x.")

matches = re.finditer(regex, text)
intrinsics = []
type_map = {
    "float": "f32",
    "double": "f64",
    "i8": "i8",
    "i16": "i16",
    "i32": "i32",
    "i64": "i64",
    "void": "()",
    "i8*": "*mut i8",
    "i16*": "*mut i16",
    "i32*": "*mut i32",
    "i64*": "*mut i64",
    "float*": "*mut f32",
    "double*": "*mut f64",
}

for match in matches:
    sig_txt = match.group(2).strip()
    sig = {}
    return_ty = type_map[re.search(".*(?= @)", sig_txt).group()]
    params = []
    for param in re.finditer("(\w+\*?)(?= %) %(\w+)", sig_txt):
        params.append(
            {
                "name": param.group(2).strip(),
                "type": type_map[param.group(1).strip()]
            }
        )

    sig["params"] = params
    sig["returns"] = return_ty

    intrinsics.append(
        {
            "name": match.group(1).strip(),
            "sig": sig,
            "description": match.group(3).strip(),
            "returns": match.group(4).strip(),
            "availability": match.group(5).strip()
        }
    )

out = open(out_filename, "w", encoding="utf8")
out.write(json.dumps(intrinsics, indent=2))
