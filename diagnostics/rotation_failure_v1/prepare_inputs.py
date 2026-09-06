"""Read the frozen Drive inputs into an external Colab cache; never write Drive."""
import argparse
import io
import json
from pathlib import Path

from diagnostic import CONDITIONS, PRODUCER, REPO, TAU, audit_inputs, write_new


def extract_source(manifest, formal):
    if formal["identity"]["expected_exact"] != PRODUCER or formal["identity"]["stage"] != "evaluation_detection":
        raise ValueError("historical source identity differs")
    ids = {e["sample_id"] for e in manifest["entries"]}
    rows = [r for r in formal["records"] if r["physical_unit_id"] in ids and r["condition"] in CONDITIONS]
    expected = {(i,c,t) for i in ids for c in CONDITIONS for t in ("negative","positive")}
    actual = {(r["physical_unit_id"],r["condition"],r["truth_role"]) for r in rows}
    if len(ids)!=100 or len(rows)!=400 or expected!=actual or any(r["threshold"]!=TAU for r in rows):
        raise ValueError("historical source coverage differs")
    return {"status":"POSTHOC_DIAGNOSTIC_ONLY","science_denominator":0,
            "source_file_id":manifest["source"]["result_file_id"],
            "identity":formal["identity"],"records":rows}


def cache_json(path, value):
    if path.exists():
        if json.loads(path.read_text())!=value:
            raise ValueError("existing input cache differs; preserve it for audit")
    else:
        write_new(path,value)


def prepare(root, fetch_bytes):
    root=Path(root).resolve()
    if root.is_relative_to(REPO):
        raise ValueError("input cache must remain outside Git worktree")
    ref=json.loads((Path(__file__).parent/"input_reference.json").read_text())
    manifest=json.loads(fetch_bytes('1lG4YYKqnim0ToDwjuEbASNXA3OXuQkUG'))
    index=json.loads(fetch_bytes('1YtBQG3gIX2TGtFDiLl5d7fnVrK3MGrI9'))
    expected=[(e['sample_id'],e['selection_stratum']) for e in ref['entries']]
    if [(e['sample_id'],e['selection_stratum']) for e in manifest['entries']]!=expected:
        raise ValueError('Drive manifest differs from frozen roster reference')
    if len(index['pairs'])!=100 or {p['sample_id'] for p in index['pairs']}!={i for i,_ in expected}:
        raise ValueError('Drive index differs from fixed roster')
    if manifest['source']['result_file_id']!='1nfqeoKjA6BlP3-JiuDsW8f9YJqZ0PsqO':
        raise ValueError('historical source file differs')
    root.mkdir(parents=True,exist_ok=True)
    cache_json(root/'manifest.json',manifest)
    cache_json(root/'drive_index.json',index)
    for pair in index['pairs']:
        for arm in ('clean','watermarked'):
            name=pair['sample_id']+'__'+arm+'.png'
            if pair[arm]['title']!=name:
                raise ValueError('pair-arm mapping differs')
            path=root/arm/name
            if not path.exists():
                data=fetch_bytes(pair[arm]['id'])
                path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as out:
                    out.write(data)
    source=extract_source(manifest,json.loads(fetch_bytes(manifest['source']['result_file_id'])))
    cache_json(root/'implementation/source_rows.json',source)
    audit=audit_inputs(root)
    if not audit['input_usable']:
        raise ValueError('image decode audit failed; retain cache and inspect it')
    return {k:v for k,v in audit.items() if k!='rows'}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    import google.auth
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseDownload
    credentials,_=google.auth.default()
    service=build('drive','v3',credentials=credentials,cache_discovery=False)
    def fetch(file_id):
        buffer=io.BytesIO()
        request=service.files().get_media(fileId=file_id)
        downloader=MediaIoBaseDownload(buffer,request)
        done=False
        while not done:
            _,done=downloader.next_chunk()
        return buffer.getvalue()
    print(json.dumps(prepare(args.root,fetch)))


if __name__=='__main__':
    main()
