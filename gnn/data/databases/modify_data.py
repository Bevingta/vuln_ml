import json
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Location of the current file
FILE = os.path.join(BASE_DIR, "complete_dataset.json")


def read_jsonl_to_array(file_path):
    data_array = []
    with open(file_path, 'r') as file:
        for line in file:
            data_array.append(json.loads(line))  # Convert the JSON string to a dictionary and append
    return data_array

def save_dataset_to_json(dataset, output_file_path):
    with open(output_file_path, 'w') as json_file:
        json.dump(dataset, json_file, indent=4)

def load_json_dataset(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)  # Load the existing array
    return data


def transform_data(input_data, db_name):
    transformed_data = []

    # Iterate over the values of the dictionary
    for entry in input_data:
            transformed_entry = {
                "idx": None,  # will redo later
                "func": entry.get("func", ""),
                "target": entry.get("target", None),
                "cwe": [entry.get("CWE ID", "")],
                "cve": entry.get("CVE ID", ""),
                "database_origin": db_name
            }
            transformed_data.append(transformed_entry)
    
    return transformed_data

'''
"idx": 410528,
        "func": "uint8_t rfc_parse_data(tRFC_MCB* p_mcb, MX_FRAME* p_frame, BT_HDR* p_buf) {\n uint8_t ead, eal, fcs;\n uint8_t* p_data = (uint8_t*)(p_buf + 1) + p_buf->offset;\n uint8_t* p_start = p_data;\n uint16_t len;\n\n if (p_buf->len < RFCOMM_CTRL_FRAME_LEN) {\n    RFCOMM_TRACE_ERROR(\"Bad Length1: %d\", p_buf->len);\n return (RFC_EVENT_BAD_FRAME);\n }\n\n  RFCOMM_PARSE_CTRL_FIELD(ead, p_frame->cr, p_frame->dlci, p_data);\n if (!ead) {\n    RFCOMM_TRACE_ERROR(\"Bad Address(EA must be 1)\");\n\n     return (RFC_EVENT_BAD_FRAME);\n   }\n   RFCOMM_PARSE_TYPE_FIELD(p_frame->type, p_frame->pf, p_data);\n  RFCOMM_PARSE_LEN_FIELD(eal, len, p_data);\n \n   p_buf->len -= (3 + !ead + !eal + 1); /* Additional 1 for FCS */\n   p_buf->offset += (3 + !ead + !eal);\n\n /* handle credit if credit based flow control */\n if ((p_mcb->flow == PORT_FC_CREDIT) && (p_frame->type == RFCOMM_UIH) &&\n (p_frame->dlci != RFCOMM_MX_DLCI) && (p_frame->pf == 1)) {\n    p_frame->credit = *p_data++;\n    p_buf->len--;\n    p_buf->offset++;\n } else\n    p_frame->credit = 0;\n\n if (p_buf->len != len) {\n    RFCOMM_TRACE_ERROR(\"Bad Length2 %d %d\", p_buf->len, len);\n return (RFC_EVENT_BAD_FRAME);\n }\n\n  fcs = *(p_data + len);\n\n /* All control frames that we are sending are sent with P=1, expect */\n /* reply with F=1 */\n /* According to TS 07.10 spec ivalid frames are discarded without */\n /* notification to the sender */\n switch (p_frame->type) {\n case RFCOMM_SABME:\n if (RFCOMM_FRAME_IS_RSP(p_mcb->is_initiator, p_frame->cr) ||\n !p_frame->pf || len || !RFCOMM_VALID_DLCI(p_frame->dlci) ||\n !rfc_check_fcs(RFCOMM_CTRL_FRAME_LEN, p_start, fcs)) {\n        RFCOMM_TRACE_ERROR(\"Bad SABME\");\n return (RFC_EVENT_BAD_FRAME);\n } else\n return (RFC_EVENT_SABME);\n\n case RFCOMM_UA:\n if (RFCOMM_FRAME_IS_CMD(p_mcb->is_initiator, p_frame->cr) ||\n !p_frame->pf || len || !RFCOMM_VALID_DLCI(p_frame->dlci) ||\n !rfc_check_fcs(RFCOMM_CTRL_FRAME_LEN, p_start, fcs)) {\n        RFCOMM_TRACE_ERROR(\"Bad UA\");\n return (RFC_EVENT_BAD_FRAME);\n } else\n return (RFC_EVENT_UA);\n\n case RFCOMM_DM:\n if (RFCOMM_FRAME_IS_CMD(p_mcb->is_initiator, p_frame->cr) || len ||\n !RFCOMM_VALID_DLCI(p_frame->dlci) ||\n !rfc_check_fcs(RFCOMM_CTRL_FRAME_LEN, p_start, fcs)) {\n        RFCOMM_TRACE_ERROR(\"Bad DM\");\n return (RFC_EVENT_BAD_FRAME);\n } else\n return (RFC_EVENT_DM);\n\n case RFCOMM_DISC:\n if (RFCOMM_FRAME_IS_RSP(p_mcb->is_initiator, p_frame->cr) ||\n !p_frame->pf || len || !RFCOMM_VALID_DLCI(p_frame->dlci) ||\n !rfc_check_fcs(RFCOMM_CTRL_FRAME_LEN, p_start, fcs)) {\n        RFCOMM_TRACE_ERROR(\"Bad DISC\");\n return (RFC_EVENT_BAD_FRAME);\n } else\n return (RFC_EVENT_DISC);\n\n case RFCOMM_UIH:\n if (!RFCOMM_VALID_DLCI(p_frame->dlci)) {\n        RFCOMM_TRACE_ERROR(\"Bad UIH - invalid DLCI\");\n return (RFC_EVENT_BAD_FRAME);\n } else if (!rfc_check_fcs(2, p_start, fcs)) {\n        RFCOMM_TRACE_ERROR(\"Bad UIH - FCS\");\n return (RFC_EVENT_BAD_FRAME);\n } else if (RFCOMM_FRAME_IS_RSP(p_mcb->is_initiator, p_frame->cr)) {\n /* we assume that this is ok to allow bad implementations to work */\n        RFCOMM_TRACE_ERROR(\"Bad UIH - response\");\n return (RFC_EVENT_UIH);\n } else\n return (RFC_EVENT_UIH);\n }\n\n return (RFC_EVENT_BAD_FRAME);\n}\n",
        "target": 1,
        "cwe": [
            "CWE-125"
        ],
        "cve": "CVE-2018-9503",
        "database_origin": "bigvul"
'''

def get_database_info(database=None, file_path=None, db_name=None):
    if database is None:
        if file_path is not None: database = load_json_dataset(file_path)
        else: raise ValueError
    print(f"Size of {db_name} dataset: {len(database)} entries.")

    vuln = safe = 0
    for i in range(len(database)):
        current_entry = database[i]
        if current_entry["target"] == 0:
            safe += 1
        else:
            vuln += 1

    print(f"Number of vulnerable entries: {vuln}")
    print(f"Number of safe entries: {safe}")

def remove_duplicates(data):
    new_data = []
    seen_funcs = {}
    duplicate_idx = []
    num_vul_duplicates = num_safe_duplicates = 0
    for i in range(len(data)):
        func = data[i]["func"]
        if func in seen_funcs:
            duplicate_idx.append(data[i]["idx"])
            if data[i]["target"] == 0: num_safe_duplicates +=1 
            else: num_vul_duplicates += 1
        else:
            seen_funcs[func] = data[i]["idx"]
            new_data.append(data[i])
    print(f"Removing {num_safe_duplicates} safe duplicates, and {num_vul_duplicates} vulnerable duplicates.")
    return new_data

def redo_idx_numbering(data):
    for i in range(len(data)):
        data[i]["idx"] = i

database_name = "devign"
database_path = os.path.join(BASE_DIR, f"{database_name}.json")
database = load_json_dataset(database_path)
transformed_dataset = transform_data(database, database_name)
redo_idx_numbering(transformed_dataset)

save_dataset_to_json(transformed_dataset, database_path)