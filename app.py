import streamlit as st
import os
from typing import List
from database   import init_db, get_user, add_user
from indexer    import scan_university_folder, build_index, get_folder_structure
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
# INIT DB + SESSION STATE
# ============================================================
init_db()

if "logged_in"    not in st.session_state:
    st.session_state.logged_in    = False
if "user"         not in st.session_state:
    st.session_state.user         = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

GROQ_API_KEY = st.secrets.get(
    "GROQ_API_KEY",
    os.environ.get("GROQ_API_KEY", "")
)

# ============================================================
# HELPER — Returns ONLY folders + files user can access
# Restricted content completely hidden — never shown
# ============================================================
def get_accessible_structure(user_access_dirs: List[str]) -> dict:
    full_structure = get_folder_structure()
    accessible     = {}

    for folder, files in full_structure.items():
        has_access = any(
            folder.startswith(allowed)
            for allowed in user_access_dirs
        )
        if has_access:
            accessible[folder] = files

    return accessible


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("🎓 University RAG")
    st.divider()

    # ── NOT LOGGED IN ──
    if not st.session_state.logged_in:
        st.subheader("🔐 Login")

        user_id  = st.text_input("User ID")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            user = get_user(user_id, password)
            if user:
                st.session_state.logged_in    = True
                st.session_state.user         = user
                st.session_state.chat_history = []
                st.rerun()
            else:
                st.error("❌ Invalid credentials")

        st.divider()
        st.caption("Sample Login Credentials:")
        st.code("""
cse_student  / pass123
ece_student  / pass123
mech_student / pass123
civil_student/ pass123
mba_student  / pass123
admin_user   / admin123
hod_cse      / hod123
principal    / prin123
        """)

    # ── LOGGED IN ──
    else:
        user = st.session_state.user

        # User info card
        st.success(f"✅ {user['user_id']}")
        st.write(f"**Role      :** {user['role']}")
        st.write(f"**Department:** {user['department']}")
        st.divider()

        # Show ONLY accessible folders + files
        # get_folder_structure() never called here
        # Restricted folders completely invisible
        st.subheader("📁 Your Documents")
        accessible_sidebar = get_accessible_structure(user["access_dirs"])

        if accessible_sidebar:
            for folder, files in accessible_sidebar.items():
                with st.expander(
                    f"📂 {folder} ({len(files)} files)"
                ):
                    if files:
                        for f in files:
                            st.write(f"  📄 {f}")
                    else:
                        st.caption("No PDFs in this folder")
        else:
            st.warning("No documents accessible")

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.logged_in    = False
            st.session_state.user         = None
            st.session_state.chat_history = []
            st.rerun()


# ============================================================
# MAIN AREA
# ============================================================
st.title("🎓 University Knowledge Assistant")

# ─────────────────────────────────────────────────────
# NOT LOGGED IN — landing page
# ─────────────────────────────────────────────────────
if not st.session_state.logged_in:
    st.info("👈 Please login from the sidebar to start.")
    st.divider()

    st.subheader("📌 How It Works")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.info("**Step 1**\n\nLogin with your university credentials")
    with col2:
        st.info("**Step 2**\n\nSystem fetches your access level from DB")
    with col3:
        st.info("**Step 3**\n\nSelect folder and PDF scope")
    with col4:
        st.info("**Step 4**\n\nAsk question — only your docs searched")

# ─────────────────────────────────────────────────────
# LOGGED IN
# ─────────────────────────────────────────────────────
else:
    user       = st.session_state.user

    # Get accessible structure ONCE — reuse everywhere
    # This is the ONLY source of file/folder info for users
    accessible = get_accessible_structure(user["access_dirs"])

    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 My Access", "⚙️ Admin"])

    # ══════════════════════════════════════════════════
    # TAB 1 — CHAT
    # ══════════════════════════════════════════════════
    with tab1:

        # Top info bar
        st.info(
            f"🔐 Logged in as **{user['user_id']}**  |  "
            f"Role: **{user['role']}**  |  "
            f"Department: **{user['department']}**"
        )

        # ─────────────────────────────────────────────
        # STEP 1 — FOLDER SELECTION
        # Only accessible folders shown in dropdown
        # First folder auto selected by default
        # ─────────────────────────────────────────────
        st.subheader("📂 Step 1 — Select Folder")

        folder_list = list(accessible.keys())

        if not folder_list:
            st.error(
                "⛔ No accessible folders found. "
                "Contact admin to get access."
            )
            st.stop()

        selected_folder = st.selectbox(
            "Choose a folder to search in:",
            options     = folder_list,
            index       = 0,  # ← first folder selected by default
            format_func = lambda x: (
                f"📂 {x}  —  {len(accessible[x])} PDF(s)"
            )
        )

        # Show PDFs inside selected folder as pill badges
        folder_files = accessible.get(selected_folder, [])
        if folder_files:
            st.caption(
                f"PDFs available in **{selected_folder}**:"
            )
            badge_cols = st.columns(min(len(folder_files), 4))
            for i, fname in enumerate(folder_files):
                badge_cols[i % 4].markdown(
                    f"<span style='"
                    f"background:#1f4e79;"
                    f"color:white;"
                    f"padding:3px 10px;"
                    f"border-radius:12px;"
                    f"font-size:12px;"
                    f"display:inline-block;"
                    f"margin:2px"
                    f"'>📄 {fname}</span>",
                    unsafe_allow_html=True
                )
        else:
            st.warning("No PDFs found in this folder")

        st.divider()

        # ─────────────────────────────────────────────
        # STEP 2 — PDF SELECTION
        # "All PDFs" default, or pick one specific PDF
        # Only PDFs from selected folder shown
        # ─────────────────────────────────────────────
        st.subheader("📄 Step 2 — Select PDF")
        st.caption(
            "Search all PDFs in the folder "
            "OR pick one specific PDF"
        )

        # Build options
        pdf_options = (
            ["🔍 All PDFs in this folder"]
            + folder_files
        )

        selected_pdf = st.selectbox(
            "Search in:",
            options = pdf_options,
            index   = 0,  # ← "All PDFs" default
        )

        # Determine scope
        if selected_pdf == "🔍 All PDFs in this folder":
            scope_type  = "folder"
            scope_value = selected_folder
            scope_label = f"All PDFs in 📂 {selected_folder}"
        else:
            scope_type  = "file"
            scope_value = selected_pdf
            scope_label = (
                f"📄 {selected_pdf}  "
                f"(in 📂 {selected_folder})"
            )

        # Show scope confirmation box
        st.success(f"🔎 Search scope set to: **{scope_label}**")

        st.divider()

        # ─────────────────────────────────────────────
        # STEP 3 — CHAT
        # ─────────────────────────────────────────────
        st.subheader("💬 Step 3 — Ask Your Question")

        # Clear chat
        col_clear, _ = st.columns([1, 6])
        with col_clear:
            if st.button("🗑️ Clear Chat"):
                st.session_state.chat_history = []
                st.rerun()

        # ── Chat History ──
        if not st.session_state.chat_history:
            st.caption(
                "💡 No messages yet. "
                "Select your scope above and ask a question!"
            )

        for chat in st.session_state.chat_history:

            with st.chat_message("user"):
                st.caption(f"🔎 Scope: {chat['scope']}")
                st.write(chat["question"])

            with st.chat_message("assistant"):
                st.write(chat["answer"])

                if chat.get("sources"):
                    with st.expander(
                        f"📎 {len(chat['sources'])} source(s) used"
                    ):
                        for src in chat["sources"]:
                            st.markdown(
                                f"📄 **{src['file']}**  |  "
                                f"📂 `{src['folder']}`"
                            )
                            st.caption(src["content"] + "...")
                            st.divider()
                else:
                    st.caption("⚠️ No matching sources found")

        # ── Question Input ──
        question = st.chat_input(
            f"Ask about {scope_label}..."
        )

        if question:

            with st.chat_message("user"):
                st.caption(f"🔎 Scope: {scope_label}")
                st.write(question)

            with st.chat_message("assistant"):
                with st.spinner(
                    f"Searching in {scope_label}..."
                ):
                    result = ask_question(
                        question    = question,
                        access_dirs = user["access_dirs"],
                        groq_api_key= GROQ_API_KEY,
                        scope_type  = scope_type,
                        scope_value = scope_value
                    )

                st.write(result["answer"])

                if result["retrieved_docs"]:
                    with st.expander(
                        f"📎 {len(result['retrieved_docs'])} "
                        f"source(s) used"
                    ):
                        for src in result["retrieved_docs"]:
                            st.markdown(
                                f"📄 **{src['file']}**  |  "
                                f"📂 `{src['folder']}`"
                            )
                            st.caption(
                                src["content"] + "..."
                            )
                            st.divider()
                else:
                    st.warning(
                        f"⚠️ No results found in "
                        f"**{scope_label}**. "
                        f"Try selecting a different "
                        f"folder or PDF."
                    )

            # Save to history
            st.session_state.chat_history.append({
                "question": question,
                "scope"   : scope_label,
                "answer"  : result["answer"],
                "sources" : result["retrieved_docs"]
            })

    # ══════════════════════════════════════════════════
    # TAB 2 — MY ACCESS
    # Only shows what user CAN access
    # Blocked/restricted folders completely hidden
    # ══════════════════════════════════════════════════
    with tab2:
        st.subheader("📊 Your Access Details")

        # User info metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("User ID",    user["user_id"])
        with col2:
            st.metric("Role",       user["role"])
        with col3:
            st.metric("Department", user["department"])

        st.divider()

        # Accessible directories list
        # NO blocked folders shown here
        st.write("### ✅ Your Accessible Folders")
        if user["access_dirs"]:
            for d in user["access_dirs"]:
                st.success(f"✅ {d}")
        else:
            st.warning("No folder access assigned to your account")

        st.divider()

        # Detailed file breakdown
        # Only accessible files shown — nothing else
        st.write("### 📁 Your Accessible Files")
        if accessible:
            total_files = sum(len(v) for v in accessible.values())
            st.caption(
                f"You have access to **{len(accessible)} folder(s)** "
                f"containing **{total_files} PDF(s)**"
            )
            st.divider()

            for folder, files in accessible.items():
                with st.expander(
                    f"📂 {folder}  —  {len(files)} file(s)",
                    expanded=True
                ):
                    if files:
                        for f in files:
                            st.write(f"  📄 {f}")
                    else:
                        st.caption("No PDFs in this folder")
        else:
            st.warning(
                "No accessible documents found. "
                "Please contact admin."
            )

    # ══════════════════════════════════════════════════
    # TAB 3 — ADMIN
    # get_folder_structure() ONLY called inside here
    # Regular users see only "access required" message
    # ══════════════════════════════════════════════════
    with tab3:

        # Hard gate — non admin sees nothing
        if user["role"] not in ["admin", "principal"]:
            st.warning(
                "⛔ Admin or Principal access required "
                "for this tab."
            )

        else:
            st.subheader("⚙️ Admin Panel")

            # ── Index Documents ──
            st.write("### 📥 Index Documents")
            st.caption(
                "Scan the university folder and "
                "index all PDFs into FAISS vector store"
            )

            col_btn, col_info = st.columns([1, 3])

            with col_btn:
                if st.button(
                    "🔄 Scan & Index",
                    use_container_width=True
                ):
                    with st.spinner(
                        "Scanning university folder..."
                    ):
                        records = scan_university_folder()

                    with st.spinner(
                        f"Indexing {len(records)} documents..."
                    ):
                        build_index(records)

                    st.success(
                        f"✅ Successfully indexed "
                        f"{len(records)} documents!"
                    )

            with col_info:
                # Admin sees full structure stats
                full_structure = get_folder_structure()
                total_files    = sum(
                    len(v) for v in full_structure.values()
                )
                st.info(
                    f"📊 University Directory Stats:  "
                    f"**{len(full_structure)} folders**  |  "
                    f"**{total_files} total PDFs**"
                )

            st.divider()

            # ── Add New User ──
            st.write("### ➕ Add New User")

            # Admin sees ALL folders for assignment
            full_structure = get_folder_structure()

            col1, col2 = st.columns(2)
            with col1:
                new_id   = st.text_input(
                    "User ID", key="new_uid"
                )
                new_pass = st.text_input(
                    "Password",
                    type="password",
                    key="new_pass"
                )
                new_role = st.selectbox(
                    "Role",
                    ["student", "faculty", "hod", "admin"],
                    key="new_role"
                )
                new_dept = st.text_input(
                    "Department", key="new_dept"
                )

            with col2:
                st.write("**Select Access Directories:**")
                st.caption(
                    "Admin assigns which folders "
                    "this user can access"
                )
                new_dirs = st.multiselect(
                    "Access Directories",
                    options = list(full_structure.keys()),
                    key     = "new_dirs"
                )

                # Preview selected access
                if new_dirs:
                    st.write("**Access Preview:**")
                    for d in new_dirs:
                        fc = len(full_structure.get(d, []))
                        st.success(
                            f"✅ {d}  ({fc} files)"
                        )

            if st.button("➕ Add User"):
                if (
                    new_id   and
                    new_pass and
                    new_dept and
                    new_dirs
                ):
                    success = add_user(
                        new_id,   new_pass,
                        new_role, new_dept,
                        new_dirs
                    )
                    if success:
                        st.success(
                            f"✅ User **'{new_id}'** "
                            f"added successfully!"
                        )
                    else:
                        st.error(
                            f"❌ User ID **'{new_id}'** "
                            f"already exists"
                        )
                else:
                    st.warning(
                        "⚠️ Please fill all fields and "
                        "select at least one directory"
                    )

            st.divider()

            # ── Full Folder Structure — Admin Only ──
            st.write(
                "### 🗂️ Complete University "
                "Folder Structure"
            )

            if full_structure:
                for folder, files in full_structure.items():
                    with st.expander(
                        f"📂 {folder}  —  "
                        f"{len(files)} file(s)"
                    ):
                        if files:
                            for f in files:
                                st.write(f"  📄 {f}")
                        else:
                            st.caption(
                                "No PDFs in this folder"
                            )
            else:
                st.warning(
                    "No folders found. Make sure the "
                    "university/ directory exists "
                    "with PDFs inside."
                )
