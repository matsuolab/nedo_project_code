import sys
import json
import urllib.request


def notify(url, text):
    data = {
        'text': text
    }
    headers = {
        'Content-Type': 'application/json'
    }
    req = urllib.request.Request(url, json.dumps(data).encode(), headers)
    with urllib.request.urlopen(req) as res:
        print(res.read().decode())


def parse_squeue(section):
    data = json.loads(section)
    process = False
    result = []
    for job in data['jobs']:
        if job['account'] == 'haijima':
            result.append(
                f'{job["job_id"]} {job["name"]} {job["job_resources"]["nodes"]} {job["group_name"]} {job["job_state"]}'
            )
            if job['name'] == 'pretrain_mistral_9b' and job['job_state'] == 'RUNNING':
                process = True
    if process is False:
        notify('https://hooks.slack.com/services/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', f'WARNING: There is no running mistral job')
    msg = '\n'.join(sorted(result, key=lambda x: x.split()[3]))
    notify('https://hooks.slack.com/services/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', f'```{msg}```')


def parse_sinfo(section):
    # 特に解析するものがなさそう？
    pass


def parse_disk(section):
    lines = section.split("\n")
    for line in lines:
        if "/storage3" in line:
            notify('https://hooks.slack.com/services/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', line)
            values = line.split()
            if len(values) >= 5:
                value = values[4]
                if value == '100%':
                    notify('https://hooks.slack.com/services/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', f'WARNING: Disk usage is 100% on /persistentshare')


git_normal = """* main
On branch main
Your branch is up to date with 'origin/main'.

nothing to commit, working tree clean
"""


def parse_git(section):
    if section != git_normal:
        notify('https://hooks.slack.com/services/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', f'WARNING: Git status is not normal\n```{section}```')


def parse_who(section):
    notify('https://hooks.slack.com/services/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', f'```{section}```')


def main():
    if len(sys.argv) != 2:
        print("Usage: python monitor.py [filename]")
        sys.exit(1)

    filename = sys.argv[1]
    print("Filename: " + filename)

    with open(filename, "r") as f:
        # =====任意の名前=====となっている単位に各ファイルのセクションを分ける
        prev = None
        sections = {}
        current_section = []
        lines = f.readlines()
        for line in lines:
            if line.startswith("=====") and line.endswith("=====\n"):
                if current_section:
                    # 初回はスキップされる
                    sections[prev] = ''.join(current_section)
                    current_section = []
                prev = line.strip()
                continue

            current_section.append(line)
        if current_section:
            sections[prev] = ''.join(current_section)

        parse_squeue(sections["=====squeue====="])
        parse_sinfo(sections["=====sinfo====="])
        parse_disk(sections["=====disk====="])
        parse_git(sections["=====git====="])
        parse_who(sections["=====who====="])


if __name__ == "__main__":
    main()
