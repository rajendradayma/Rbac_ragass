import streamlit as st
import os
from database import init_db, get_user, add_user
from indexer  import scan_university_folder, build_index, get_folder_structure
from rag_engine import ask_question

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title = "University RAG System",
    page_icon  = "🎓",
    layout     = "wide"
)

# ============================================================
# INIT
# ============================================================
init_db()

if "logged_in"   not in st.session_state:
    st.session_state.logged_in   = False
if "user"        not in st.session_state:
    st.session_state.user        = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎓 University RAG")
    st.divider()

    # Login / Logout
    if not st.session_state.logged_in:
        st.subheader("🔐 Login")
        user_id  = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            user = get_user(user_id, password)
            if user:
                st.session_state.logged_in = True
                st.session_state.user      = user
                st.success(f"Welcome {user['user_id']}!")
                st.rerun()
            else:
                st.error("Invalid credentials")

        st.divider()
        st.caption("Sample Users:")
        st.code("""
cse_student  / pass123
ece_student  / pass123
mech_student / pass123
admin_user   / admin123
principal    / prin123
        """)

    else:
        user = st.session_state.user
        st.success(f"✅ {user['user_id']}")
        st.write(f"**Role:** {user['role']}")
        st.write(f"**Dept:** {user['department']}")

        st.divider()
        st.subheader("📁 Your Access")
        for d in user["access_dirs"]:
            st.write(f"  📂 `{d}`")

        st.divider()

        # Folder structure viewer
        with st.expander("🗂️ Browse All Folders"):
            structure = get_folder_structure()
            for folder, files in structure.items():
                # Highlight accessible folders
                accessible = any(
                    folder.startswith(a)
                    for a in user["access_dirs"]
                )
                icon = "🔓" if accessible else "🔒"
                st.write(f"{icon} **{folder}**")
                for f in files:
                    color = "green" if accessible else "red"
                    st.markdown(
                        f"&nbsp;&nbsp;&nbsp;📄 :{color}[{f}]",
                        unsafe_allow_html=True
                    )

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in    = False
            st.session_state.user         = None
            st.session_state.chat_history = []
            st.rerun()

# ============================================================
# MAIN AREA
# ============================================================
st.title("🎓 University Knowledge Assistant")

if not st.session_state.logged_in:
    st.info("Please login from the sidebar to start asking questions.")

    # Show system overview
    st.subheader("📌 System Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1**\nLogin with your university credentials")
    with col2:
        st.info("**Step 2**\nSystem fetches your access from DB")
    with col3:
        st.info("**Step 3**\nAsk questions — only your docs retrieved")

else:
    user = st.session_state.user

    # Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 My Access", "⚙️ Admin"])

    # ──────────────────────────────────────────
    # TAB 1 — CHAT
    # ──────────────────────────────────────────
    with tab1:

        # Show access info
        st.info(
            f"🔐 Logged in as **{user['user_id']}** | "
            f"Access: **{', '.join(user['access_dirs'])}**"
        )

        # Chat history display
        for chat in st.session_state.chat_history:
            with st.chat_message("user"):
                st.write(chat["question"])
            with st.chat_message("assistant"):
                st.write(chat["answer"])
                if chat.get("sources"):
                    with st.expander("📎 Sources Retrieved"):
                        for src in chat["sources"]:
                            st.write(f"📄 **{src['file']}** | 📂 `{src['folder']}`")
                            st.caption(src["content"] + "...")

        # Question input
        question = st.chat_input("Ask a question...")

        if question:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching authorized documents..."):

                    # CORE: get access from DB → pass to RAG
                    result = ask_question(
                        question    = question,
                        access_dirs = user["access_dirs"],  # ← FROM DB
                        groq_api_key= GROQ_API_KEY
                    )

                st.write(result["answer"])

                if result["retrieved_docs"]:
                    with st.expander(
                        f"📎 {len(result['retrieved_docs'])} Sources Used"
                    ):
                        for src in result["retrieved_docs"]:
                            st.write(f"📄 **{src['file']}** | 📂 `{src['folder']}`")
                            st.caption(src["content"] + "...")
                else:
                    st.warning("No authorized documents found for this query.")

            # Save to history
            st.session_state.chat_history.append({
                "question": question,
                "answer"  : result["answer"],
                "sources" : result["retrieved_docs"]
            })

    # ──────────────────────────────────────────
    # TAB 2 — MY ACCESS
    # ──────────────────────────────────────────
    with tab2:
        st.subheader("📊 Your Access Details")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("User ID",    user["user_id"])
            st.metric("Role",       user["role"])
            st.metric("Department", user["department"])

        with col2:
            st.write("**Accessible Directories:**")
            for d in user["access_dirs"]:
                st.success(f"✅ {d}")

            # Show what's blocked
            structure    = get_folder_structure()
            blocked_dirs = [
                f for f in structure.keys()
                if not any(
                    f.startswith(a)
                    for a in user["access_dirs"]
                )
            ]
            if blocked_dirs:
                st.write("**Blocked Directories:**")
                for d in blocked_dirs:
                    st.error(f"❌ {d}")

    # ──────────────────────────────────────────
    # TAB 3 — ADMIN
    # ──────────────────────────────────────────
    with tab3:
        if user["role"] != "admin" and user["role"] != "principal":
            st.warning("Admin access required")
        else:
            st.subheader("⚙️ Admin Panel")

            # Index documents
            st.write("### 📥 Index Documents")
            if st.button("🔄 Scan & Index University Folder"):
                with st.spinner("Scanning and indexing..."):
                    records = scan_university_folder()
                    build_index(records)
                st.success(f"Indexed {len(records)} documents!")

            st.divider()

            # Add new user
            st.write("### ➕ Add New User")
            col1, col2 = st.columns(2)
            with col1:
                new_id   = st.text_input("User ID")
                new_pass = st.text_input("Password", type="password")
                new_role = st.selectbox(
                    "Role",
                    ["student","faculty","hod","admin"]
                )
            with col2:
                new_dept = st.text_input("Department")
                structure= get_folder_structure()
                new_dirs = st.multiselect(
                    "Access Directories",
                    options=list(structure.keys())
                )

            if st.button("Add User"):
                if new_id and new_pass and new_dirs:
                    success = add_user(
                        new_id, new_pass,
                        new_role, new_dept, new_dirs
                    )
                    if success:
                        st.success(f"User '{new_id}' added!")
                    else:
                        st.error("User ID already exists")

            st.divider()

            # Folder structure
            st.write("### 🗂️ Indexed Folder Structure")
            structure = get_folder_structure()
            for folder, files in structure.items():
                with st.expander(f"📂 {folder} ({len(files)} files)"):
                    for f in files:
                        st.write(f"  📄 {f}")
