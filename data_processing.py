import json
import csv
from dateutil.parser import parse

def read_data():
    # 读取工人属性
    worker_quality = {}
    with open("worker_quality.csv", "r", encoding="utf-8") as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)  # 跳过标题行
        for line in csvreader:
            if float(line[1]) > 0.0:
                worker_quality[int(line[0])] = float(line[1]) / 100.0

    # 读取项目 ID
    with open("project_list.csv", "r", encoding="utf-8") as file:
        project_list_lines = file.readlines()

    project_dir = "project/"
    entry_dir = "entry/"
    all_begin_time = parse("2018-01-01T0:0:0Z")

    project_info = {}
    entry_info = {}
    limit = 24
    industry_list = {}
    for line in project_list_lines:
        line = line.strip('\n').split(',')
        project_id = int(line[0])
        entry_count = int(line[1])

        with open(project_dir + f"project_{project_id}.txt", "r", encoding="utf-8") as file:
            htmlcode = file.read()
        text = json.loads(htmlcode)

        project_info[project_id] = {}
        project_info[project_id]["sub_category"] = int(text["sub_category"])
        project_info[project_id]["category"] = int(text["category"])
        project_info[project_id]["entry_count"] = int(text["entry_count"])
        project_info[project_id]["start_date"] = parse(text["start_date"])
        project_info[project_id]["deadline"] = parse(text["deadline"])

        if text["industry"] not in industry_list:
            industry_list[text["industry"]] = len(industry_list)
        project_info[project_id]["industry"] = industry_list[text["industry"]]

        entry_info[project_id] = {}
        k = 0
        while k < entry_count:
            try:
                with open(entry_dir + f"entry_{project_id}_{k}.txt", "r", encoding="utf-8") as file:
                    htmlcode = file.read()
                text = json.loads(htmlcode)

                for item in text["results"]:
                    entry_number = int(item["entry_number"])
                    entry_info[project_id][entry_number] = {}
                    entry_info[project_id][entry_number]["entry_created_at"] = parse(item["entry_created_at"])
                    entry_info[project_id][entry_number]["worker"] = int(item["author"])
            except (FileNotFoundError, json.JSONDecodeError):
                pass
            k += limit

    return worker_quality, project_info, entry_info