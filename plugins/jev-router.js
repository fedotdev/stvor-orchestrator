/**
 * jev-router: OpenCode plugin exposing one tool, `jev_route`.
 *
 * The tool runs the local `agent_router` harness, which asks Jev (TypeSafe
 * System One) a fixed set of typed questions and derives a RoutingPlan in plain
 * Python. The tool classifies only: it has no write access, no shell surface of
 * its own and hands nothing executable back to the model. The harness decides
 * what is applied; the model never interprets the plan.
 *
 * Safety properties:
 *  - the request text is passed through a temp file, never interpolated into a
 *    shell string, and is scrubbed before it reaches the artifact;
 *  - the API key is read from the environment by the harness and never passed
 *    through the tool arguments or the model context;
 *  - `observe` is the default mode and cannot change agent behaviour.
 */
import { tool } from "@opencode-ai/plugin";
import { mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const PLUGIN_DIR = dirname(fileURLToPath(import.meta.url));
// agent_router/ lives directly under the OpenCode config root.
const ROUTER_ROOT = dirname(PLUGIN_DIR);
const HARD_TIMEOUT_MS = 60_000;

export const plugin = async (input) => {
    const shell = input.$;
    const projectDirectory = input.directory || process.cwd();

    return {
        tool: {
            jev_route: tool({
                description:
                    "Classify a request with Jev (TypeSafe System One) and return a deterministic RoutingPlan " +
                    "(fast_readonly | diagnostic_readonly | orchestrated_readonly | orchestrated_write | " +
                    "manual_orchestrator | clarification_required) plus an audit artifact path. " +
                    "Classification only: no file writes, no shell, no tools. Use it before starting a plan; " +
                    "obey the returned route, never reinterpret it. Default mode is observe, which cannot change behaviour.",
                args: {
                    task: tool.schema.string().trim().min(1).max(20000),
                    mode: tool.schema
                        .enum(["off", "observe", "assist", "enforce"])
                        .default("observe")
                        .describe("off: do not call Jev; observe: classify and record only; assist: propose and wait for the user; enforce: apply only the safe fast_readonly route"),
                    projectRoot: tool.schema
                        .string()
                        .optional()
                        .describe("Project root that owns .agent/artifacts; defaults to the current workspace"),
                },
                async execute(args) {
                    const root = args.projectRoot || projectDirectory;
                    const startedAt = Date.now();
                    let scratch;
                    try {
                        scratch = await mkdtemp(join(tmpdir(), "jev-route-"));
                        const taskFile = join(scratch, "task.txt");
                        await writeFile(taskFile, args.task, "utf-8");

                        const exec = shell`python -m agent_router route --task-file ${taskFile} --project-root ${root} --mode ${args.mode} --json --quiet`
                            .cwd(root)
                            .env({ PYTHONPATH: ROUTER_ROOT })
                            .quiet()
                            .nothrow();
                        let timeout;
                        const result = await Promise.race([
                            exec,
                            new Promise((_, reject) => {
                                timeout = setTimeout(
                                    () => reject(new Error(`hard timeout after ${HARD_TIMEOUT_MS}ms`)),
                                    HARD_TIMEOUT_MS,
                                );
                            }),
                        ]).finally(() => {
                            if (timeout) clearTimeout(timeout);
                        });

                        const stdout = result.text("utf-8") ?? "";
                        const stderr = result.stderr.toString("utf-8");
                        if (result.exitCode !== 0 && !stdout.trim()) {
                            return `jev_route failed (exit ${result.exitCode}): ${stderr.slice(0, 500)}`;
                        }

                        const payload = JSON.parse(stdout);
                        const plan = payload.plan || {};
                        return [
                            `route=${plan.route}`,
                            `applied=${payload.applied_route}`,
                            `mode=${payload.mode}`,
                            `run_id=${payload.run_id}`,
                            `artifact=${payload.artifact || "not written"}`,
                            `reason=${plan.reason}`,
                            `latency_ms=${payload.jev?.latency_ms ?? "n/a"} (harness ${Date.now() - startedAt}ms)`,
                            `agents=${[
                                plan.run_scout && "scout",
                                plan.run_researcher && "researcher",
                                plan.run_test_analyst && "test_analyst",
                                plan.run_implementer && "implementer",
                                plan.run_deep_reviewer && "deep_reviewer",
                            ]
                                .filter(Boolean)
                                .join(",") || "none"}`,
                            `needs_user_confirmation=${plan.needs_user_confirmation}`,
                        ].join("\n");
                    } catch (error) {
                        // Never let a router failure break the agent.
                        return `jev_route error: ${error instanceof Error ? error.message : String(error)}`;
                    } finally {
                        if (scratch) {
                            await rm(scratch, { recursive: true, force: true }).catch(() => {});
                        }
                    }
                },
            }),
        },
    };
};
