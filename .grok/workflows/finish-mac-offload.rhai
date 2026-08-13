let meta = #{
    name: "finish-mac-offload",
    description: "Finish Linux-to-Mac Thunderbolt offload: remaining code, verify, then hardware gates",
    when_to_use: "Push OdinLink Mac RAM / PyTorch / DMA work through to completion",
    phases: [
        #{ title: "Survey", detail: "inventory what is done vs still blocked" },
        #{ title: "Implement", detail: "close remaining code gaps" },
        #{ title: "Verify", detail: "build, review, fail closed on regressions" },
        #{ title: "Hardware", detail: "user confirms cable, Mac, Thunderbolt IP" },
        #{ title: "Report", detail: "write the completion report" },
    ],
};

let survey_schema = #{
    "type": "object",
    "properties": #{
        "summary": #{ "type": "string" },
        "code_gaps": #{
            "type": "array",
            "maxItems": 6,
            "items": #{ "type": "string" },
        },
        "hardware_gaps": #{
            "type": "array",
            "maxItems": 6,
            "items": #{ "type": "string" },
        },
    },
};

let verify_schema = #{
    "type": "object",
    "properties": #{
        "ok": #{ "type": "boolean" },
        "notes": #{ "type": "string" },
    },
};

let skip_hw = false;
if args != () {
    if args.skip_hardware != () {
        skip_hw = args.skip_hardware == true;
    }
}

phase("Survey");
let survey_prompt = "";
survey_prompt += "You are surveying the OdinLink-Five repo at the workspace root. ";
survey_prompt += "Use read_file, grep, and list_dir. Do not invent files. ";
survey_prompt += "Read at least: docs/PYTORCH.md, docs/HARDWARE.md, mac/README.md, ";
survey_prompt += "odinlink/remote.py, driver/odl_tb5_service.c (bind_any), ";
survey_prompt += "mac/kext/OdinLinkNHI.cpp, .github/workflows/build.yml. ";
survey_prompt += "Return code_gaps: remaining SOFTWARE work that can be done without a Mac or TB cable ";
survey_prompt += "(empty array only after you read those files and confirm the items are already implemented). ";
survey_prompt += "Return hardware_gaps: work that needs the user (cable, Thunderbolt Bridge IP, ";
survey_prompt += "ThunderboltM4 on Z690 Carbon or use MS-S1 MAX, SIP-off kext load, odl_rdma_client -a). ";
survey_prompt += "summary: two sentences on current state.";

let survey = agent(survey_prompt, #{
    label: "survey-gaps",
    capability_mode: "read-only",
    output_schema: survey_schema,
});

let code_gaps = [];
let hw_gaps = [];
let survey_summary = "survey unavailable";
if survey != () && survey.success && survey.output != () {
    if survey.output.summary != () { survey_summary = survey.output.summary; }
    if survey.output.code_gaps != () { code_gaps = survey.output.code_gaps; }
    if survey.output.hardware_gaps != () { hw_gaps = survey.output.hardware_gaps; }
}
log("survey: " + survey_summary);
log("code_gaps=" + code_gaps.len().to_string() + " hardware_gaps=" + hw_gaps.len().to_string());

phase("Implement");
let impl_prompt = "";
impl_prompt += "Implement remaining SOFTWARE gaps for Linux Ubuntu + PyTorch using Mac RAM over Thunderbolt ";
impl_prompt += "in this OdinLink-Five repo. Use read_file/grep before editing. Keep changes small and compiling. ";
impl_prompt += "Already shipped (do not redo unless broken): odinlink.RemoteStore (TB-IP bridge), ";
impl_prompt += "examples/pytorch_mac_offload.py, bind_any + skip_login in the driver, Mac kext DART buffer + ";
impl_prompt += "XDomain watch/claim, scripts/tb-hw-check.sh, docs/PYTORCH.md and docs/HARDWARE.md, CI kext+odl_tensor_send. ";
impl_prompt += "Survey summary: ";
impl_prompt += json_encode(survey_summary);
impl_prompt += " Code gaps JSON: ";
impl_prompt += json_encode(code_gaps);
impl_prompt += " Prefer: (1) any missing RemoteStore/docs wiring, (2) Mac kext login/response if still incomplete, ";
impl_prompt += "(3) Linux bind_any/skip_login robustness, (4) CI if a new file would fail the workflow. ";
impl_prompt += "Do not buy hardware or wait for a cable. Do not force-push. If nothing is left, make no edits and say so.";

let impl = agent(impl_prompt, #{
    label: "implement-gaps",
    capability_mode: "all",
});
let impl_ok = impl != () && impl.success;
if impl_ok {
    log("implement agent finished");
} else {
    log("implement agent failed or empty — continuing to verify");
}

phase("Verify");
let build_prompt = "";
build_prompt += "Verify the OdinLink-Five workspace still builds the userspace pieces this work added. ";
build_prompt += "Run: python3 -c \"from odinlink.remote import RemoteStore, _parse_url; ";
build_prompt += "assert _parse_url('tb5://10.0.0.2')[0]=='10.0.0.2'\". ";
build_prompt += "If build/ exists, cmake --build build --target odl_tensor_send odl_tb5_cli (do not rebuild the kernel unless cheap). ";
build_prompt += "Read .github/workflows/build.yml and confirm odl_tensor_send and both kext .cpp files are compiled. ";
build_prompt += "Set ok=true only if the python check passed and you inspected CI. notes: what you ran and results.";

let review_prompt = "";
review_prompt += "Adversarial review of the Mac offload path. Read odinlink/remote.py, bridge/tb_bridge_client.py, ";
review_prompt += "driver/odl_tb5_service.c bind_any/skip_login, mac/kext/OdinLinkNHI.cpp. ";
review_prompt += "Set ok=true only if you found no crash/data-loss bug in those files. ";
review_prompt += "notes: one paragraph. Do not claim DMA works without evidence of READY+rx_done on real hardware.";

let vjobs = [];
vjobs.push(#{
    prompt: build_prompt,
    label: "verify-build",
    capability_mode: "execute",
    output_schema: verify_schema,
});
vjobs.push(#{
    prompt: review_prompt,
    label: "verify-review",
    capability_mode: "read-only",
    output_schema: verify_schema,
});
let vres = parallel(vjobs);

let build_ok = false;
let review_ok = false;
let build_notes = "not run";
let review_notes = "not run";
if vres.len() > 0 && vres[0] != () && vres[0].success && vres[0].output != () {
    if vres[0].output.ok == true { build_ok = true; }
    if vres[0].output.notes != () { build_notes = vres[0].output.notes; }
}
if vres.len() > 1 && vres[1] != () && vres[1].success && vres[1].output != () {
    if vres[1].output.ok == true { review_ok = true; }
    if vres[1].output.notes != () { review_notes = vres[1].output.notes; }
}
log("verify build_ok=" + build_ok.to_string() + " review_ok=" + review_ok.to_string());

phase("Hardware");
if !skip_hw {
    let gate = "";
    gate += "Hardware gate (resume when done, or pass skip_hardware=true to skip):\n";
    gate += "1) TB cable up: Linux thunderbolt0/en05 or Mac Thunderbolt Bridge has an IP.\n";
    gate += "2) Trainer is a box with an Intel NHI (MS-S1 MAX already has TB5; this Z690 Carbon needs ThunderboltM4).\n";
    gate += "3) Mac: python3 bridge/tb_bridge_server.py --bind 0.0.0.0 --max-gb 96\n";
    gate += "4) Linux: python3 examples/pytorch_mac_offload.py --url tb5://<mac-ip>\n";
    gate += "Optional DMA: Linux insmod odl_tb5.ko skip_login=1; Mac SIP off, load kext, odl_rdma_client -a.";
    await_user("user", gate);
}

phase("Report");
let report = "";
report += "# finish-mac-offload report\n\n";
report += "## Survey\n" + survey_summary + "\n\n";
report += "## Code gaps (" + code_gaps.len().to_string() + ")\n";
for g in code_gaps { report += "- " + g + "\n"; }
report += "\n## Hardware gaps (" + hw_gaps.len().to_string() + ")\n";
for g in hw_gaps { report += "- " + g + "\n"; }
report += "\n## Implement success: " + impl_ok.to_string() + "\n";
report += "## Verify build: " + build_ok.to_string() + " — " + build_notes + "\n";
report += "## Verify review: " + review_ok.to_string() + " — " + review_notes + "\n";
report += "\nPyTorch path today: from odinlink import RemoteStore; Mac RAM via Thunderbolt IP (bridge/).\n";
report += "DMA into Mac pages is still experimental (skip_login + kext -a).\n";

let report_path = write_scratch_file("report.md", report);
complete(#{
    path: report_path,
    summary: survey_summary,
    code_gaps: code_gaps.len(),
    hardware_gaps: hw_gaps.len(),
    implement_ok: impl_ok,
    build_ok: build_ok,
    review_ok: review_ok,
    skip_hardware: skip_hw,
});
