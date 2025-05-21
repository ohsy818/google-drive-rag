# Google Drive RAG System

문서 기반 RAG(Retrieval-Augmented Generation) 시스템입니다. 이 시스템은 Office 문서(Word, PowerPoint, Excel)와 PDF 파일을 처리하여 질의응답이 가능한 AI 시스템을 구축합니다.

## 설치 방법

1. 필요한 패키지 설치:
```bash
pip install -r requirements.txt
```

2. 환경 변수 설정:
`.env` 파일을 생성하고 다음 내용을 추가하세요:
```
OPENAI_API_KEY=your_openai_api_key
SUPABASE_URL=your_supabase_project_url
SUPABASE_KEY=your_supabase_service_role_key
```

3. Google Drive API 설정 (Google Drive 통합을 사용하는 경우):
- [Google Cloud Console](https://console.cloud.google.com)에서 새 프로젝트 생성
- Google Drive API 활성화
- OAuth 2.0 클라이언트 ID 생성 (데스크톱 앱 유형)
- 생성된 credentials.json 파일을 프로젝트 디렉토리에 저장

4. Supabase 설정:
- Supabase 프로젝트에서 pgvector 확장을 활성화
- SQL 에디터에서 다음 쿼리 실행:
```sql
-- pgvector 확장 활성화
CREATE EXTENSION IF NOT EXISTS vector;

-- documents 테이블 생성
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    content TEXT,
    metadata JSONB,
    embedding VECTOR(1536)
);

-- 벡터 검색 함수 생성
create or replace function match_documents(
  query_embedding vector(1536),
  filter jsonb default '{}'::jsonb,
  match_count int default 5
)
returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
)
language sql
as $$
  select
    documents.id,
    documents.content,
    documents.metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where documents.metadata @> filter
  order by documents.embedding <=> query_embedding
  limit match_count;
$$;
```

## 사용 방법

1. 로컬 문서 처리:
```bash
python main.py process --input_dir /path/to/documents
uv run main.py process --input_dir /path/to/documents
```

2. Google Drive 문서 처리:
```bash
python main.py process-drive --folder_id <folder_id> --credentials /path/to/credentials.json
uv run main.py process-drive --folder_id 1q_3S1J2cDl9Ztc7QGL7TAoK5nP2nJBuF --credentials ./credentials.json
```
- `folder_id`는 Google Drive 폴더의 ID입니다 (URL에서 찾을 수 있음)
- 첫 실행 시 브라우저에서 Google 계정 인증이 필요합니다

3. 질의응답:
```bash
# 모든 문서 검색
python main.py query --question "your question here"
uv run main.py query --question "your question here"

# 특정 저장소의 문서만 검색
python main.py query --question "your question here" --storage_type GoogleDrive
```

## 지원하는 문서 형식

### 로컬 문서:
- Microsoft Word (.doc, .docx)
- Microsoft PowerPoint (.ppt, .pptx)
- Microsoft Excel (.xls, .xlsx)
- PDF (.pdf)
- Text (.txt)

### Google Drive 문서:
- Google Docs
- Google Sheets
- Google Slides
- Microsoft Office 문서
- PDF 파일
- Text 파일

## 주의사항

- OpenAI API 키와 Supabase 자격증명을 안전하게 관리하세요.
- Google Drive API 자격증명 파일(credentials.json)을 안전하게 보관하세요.
- 대용량 문서 처리 시 시스템 자원 사용량에 주의하세요.
- 문서 접근 권한을 적절히 설정하세요. 