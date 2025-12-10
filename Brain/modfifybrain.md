1. Add bi-directional communication

Each brain should be both listener and connector

1.1 Goals

Keep existing “brain as client” behaviour working without changes for current environments.

Extend brain so it can:

accept incoming connections from other processes (brains, envs, tools),

initiate outgoing connections to other processes,

use a single event-based messaging abstraction for both directions.

Avoid breaking current modules that assume a single connection to one env.

1.2 High level design

Introduce a ConnectionManager abstraction inside the brain process.

ConnectionManager owns:

zero or one Listener socket for incoming peers,

zero or more outgoing Client connections,

a registry mapping peer_id to a connection object.

The brain’s main loop does not talk directly to multiprocessing.connection.Client or Listener anymore. It reads and writes only via ConnectionManager.

Existing “single connection to env” behaviour is implemented as:

one outgoing connection with a default peer_id (for example "env_main"),

optional disabled listener.

1.3 Message model

Keep existing message types (observation, reward, action, terminal, control messages) unchanged.

Add optional envelope fields that ConnectionManager adds or strips:

from_id: identifier of the sending node (string).

to_id: identifier of intended recipient (string or null for broadcast).

Old code that does not care about routing can ignore these fields.

1.4 Detailed tasks for Cursor

Introduce configuration for network role

Add a simple config structure for the brain process:

brain_id: str

listen_address: Optional[(host, port)]

peers: List[(peer_id, host, port)] for outgoing connections

enable_listener: bool (default False to keep current behaviour)

Keep defaults such that if config is omitted, current “client-only” behaviour remains.

Implement ConnectionManager

Responsibilities:

Optionally create and own a Listener at listen_address and accept incoming connections.

Create outgoing Client connections to configured peers on startup and on reconnect.

Maintain a dictionary connections: Dict[peer_id, Connection] for both incoming and outgoing peers.

Provide non-blocking API:

poll_events() -> List[(peer_id, message_dict)]

send(peer_id, message_dict)

broadcast(message_dict) (optional).

Implement reconnect logic for outgoing peers:

On send/recv failure for a peer, mark it as disconnected and retry connection after a configurable backoff.

Implement accept loop for incoming peers:

Accept new connections.

Assign them a peer_id (either from handshake message or auto-generated).

Store them in connections.

Integrate ConnectionManager with the brain main loop

Replace direct Client usage in the brain with:

events = connection_manager.poll_events() inside the main loop.

connection_manager.send(target_peer_id, message) when emitting actions or control messages.

For backwards compatibility:

Provide a simple path where only one outgoing connection exists (peer_id="env").

Update existing code to use that peer_id but otherwise keep logic unchanged.

Keep existing behaviour by default

By default, config:

enable_listener = False

peers = [("env", host, port)]

This ensures current working modules continue to function as before.

Only when enable_listener=True will brain accept additional inbound connections and act as both listener and connector.

Tests and validation

Test existing env + brain setup still works unchanged with config defaults.

Test new scenario:

Start one brain with listener enabled.

Start a second brain or tool that connects as client.

Verify both directions:

brain1 can send messages to brain2 via ConnectionManager.

brain2 can send messages to brain1.

2. Continuous brain runtime with hot-swappable algorithms

Clear separation between state and algorithms

2.1 Goals

Brain process runs continuously without restart.

Algorithms (policy, memory, neuromodulators, deep nets, etc.) can be swapped or extended at runtime.

Clean separation:

Persistent state: nets, memory, neuromodulator levels, internal variables.

Algorithm modules: code that reads/writes state and produces actions.

Keep existing behaviour by wrapping current logic as a default module.

2.2 High level design

Introduce a BrainRuntime core object, which:

owns BrainState (persistent data),

owns a set of BrainModule instances (algorithmic components),

handles messaging and ticking,

exposes control operations: load, unload, replace modules.

Existing code that currently does “brain logic” is moved into one LegacyCoreModule that implements BrainModule. This preserves current behaviour as baseline.

2.3 Data structures

BrainState

Container for persistent data, not logic.

Example fields:

sensors, last_actions, cum_reward.

neuromodulators (dopamine, serotonin, etc.).

long_term_memory.

network_weights and model references.

Provides:

typed getters/setters for important parts of state,

serialization hooks if needed later.

BrainModule interface

Define a small abstract interface:

on_event(event: dict, state: BrainState) -> Optional[list[dict]]
(events can be observation, reward, terminal, control, etc.)

tick(dt: float, state: BrainState) -> Optional[list[dict]]
(continuous updates, optional).

save_state(state: BrainState) -> dict
(return module-specific state if needed).

load_state(state: BrainState, module_state: dict)
(restore module-specific state).

Modules return zero or more outgoing messages to be sent via ConnectionManager.

BrainRuntime

Holds:

state: BrainState

modules: Dict[str, BrainModule]

Methods:

register_module(name: str, module: BrainModule)

unregister_module(name: str)

replace_module(name: str, module: BrainModule) with optional state transfer.

handle_event(event: dict, source_peer: str):

calls on_event on all modules or on specific module, collects outgoing messages.

tick(dt: float):

calls tick(dt, state) on all modules, collects outgoing messages.

Only BrainRuntime knows about ConnectionManager. Modules do not deal with sockets.

2.4 Control channel for hot-swapping

New control messages sent to the brain (through any connection):

{"type": "control/load_module", "module_path": "my_pkg.new_policy", "class_name": "NewPolicyModule", "slot": "policy"}

{"type": "control/unload_module", "slot": "policy"}

{"type": "control/replace_module", "module_path": ..., "class_name": ..., "slot": "policy"}

BrainRuntime:

uses importlib.import_module to load module code,

instantiates the class,

optionally transfers existing module state:

old_state = old_module.save_state(brain_state)

new_module.load_state(brain_state, old_state)

swaps module in modules[slot].

This enables:

swapping policies,

adding or removing auxiliary modules (e.g. intrinsic motivation, exploration, logging)
without restarting the brain process.

2.5 Migration of existing code

Wrap current behaviour

Implement LegacyCoreModule that:

embeds existing brain step logic in on_event / tick,

reads/writes from BrainState instead of its own globals.

Register it as modules["core"] in BrainRuntime at startup.

Replace direct event handling

Current top-level “brain” code that processes messages and computes actions is replaced with:

runtime.handle_event(event, source_peer) for each incoming message,

per-loop runtime.tick(dt).

Keep public behaviour the same

Ensure that the sequence of outgoing action messages produced by LegacyCoreModule matches the old behaviour when no new modules are loaded.

Add new modules only when control messages are explicitly sent.

2.6 Tests and validation

Test 1: baseline

With only LegacyCoreModule registered, confirm behaviour is identical to the previous code for the same event sequence.

Test 2: dynamic module load

Load a simple dummy module at runtime that logs events or modifies a small part of state.

Confirm that the brain process stays alive, and that actions are still produced.

Test 3: policy swap

Swap LegacyCoreModule with a new policy module while brain is running.

Confirm that state persists (e.g. cumulative reward, neuromodulator levels).

3. Chat AI interface as a speech module

Brain can interact with humans and update state

3.1 Goals

Add a “speech / dialogue” module to the brain that:

uses an external LLM (ChatGPT, Gemini, or other) to handle language,

uses relevant parts of BrainState as context,

updates BrainState based on conversational results (e.g. new goals, memory updates, motivational shifts),

can both:

respond to human-initiated messages,

proactively initiate interaction based on internal state or triggers.

Integration must be modular and must not break existing logic when disabled.

3.2 High level design

Implement ChatInterfaceModule as a BrainModule.

Implement an LLMClient abstraction so the module is independent of provider and API details.

Define new message types for human interaction:

{"type": "human_input", "content": "...", "channel": "speech"} to brain.

{"type": "speech_output", "content": "...", "target": "human"} from brain.

3.3 LLMClient abstraction

Interface:

generate_reply(prompt: str, history: list[dict], tools: Optional[list] = None) -> str

Implementation details:

Provide at least one implementation for local development:

either a stub that echoes input,

or a simple call to an external LLM API (provider chosen by configuration).

ChatInterfaceModule should get an LLMClient instance via dependency injection or configuration.

3.4 ChatInterfaceModule behaviour

Inputs and outputs

Listens for:

human_input events from connections dedicated to human UIs,

internal events that indicate a need to speak (for example internal_alert, high curiosity, new discovery).

Emits:

speech_output events, which the runtime routes to human-facing connection(s),

optional state-update events, or directly modifies BrainState.

Context construction

On each interaction, the module constructs an LLM prompt using:

recent human messages (stored in its own conversation history),

recent brain actions or decisions,

key fields from BrainState, such as:

current goals,

task or environment description,

important long-term memories relevant to the user,

current neuromodulator / motivational summary (as text).

The mapping from BrainState to text is deterministic and rule-based, so it can be improved over time without changing the rest.

Updating brain state from LLM responses

Module parses the LLM reply and:

optionally writes structured data into BrainState:

new goal descriptions,

updated task priorities,

updated attitudes or preferences,

notes in long-term memory.

For safety and robustness, use a simple, explicit output format where possible:

instruct LLM to respond with both text (for human) and plan or state_update in a parseable block.

This design is high-level in the PRD; exact prompt and parsing strategy can be refined later.

Proactive behaviour

ChatInterfaceModule.tick(dt, state):

monitors timing and state to decide if the brain should initiate speech.

Example triggers:

long period without human interaction but high uncertainty or confusion in state,

significant internal state change (new hypothesis, major reward change),

explicit “request_human_help” flag set by other modules.

When triggered, it composes a “thought” request to the LLM, gets a reply, and emits a speech_output event.

3.5 Integration with runtime and connections

New connection roles:

A human UI client (e.g. another process, web UI, CLI) connects to the brain via ConnectionManager with a specific peer_id (for example "human_ui").

Message flow:

Human UI → brain:

sends {"type": "human_input", "content": "...", "source": "user", "channel": "speech"}.

ConnectionManager passes event to BrainRuntime, which forwards it to all modules, including ChatInterfaceModule.

ChatInterfaceModule:

runs the LLM,

updates BrainState,

returns a speech_output message in its on_event response.

Brain runtime:

routes speech_output back via ConnectionManager.send("human_ui", message).

3.6 Configuration and safety

Provide configuration options for:

enable_chat_module: bool (default False to not affect current behaviour).

llm_provider, API keys, model name, max tokens, rate limits.

max frequency of proactive messages (to avoid spam).

When disabled, ChatInterfaceModule is not registered and no LLM calls are made.

3.7 Tests and validation

Test 1: chat disabled

With enable_chat_module=False, confirm that brain works as before and no LLM calls are attempted.

Test 2: single interaction

With enable_chat_module=True and a stub LLM implementation:

send a human_input event,

confirm a speech_output is produced and routed back to the human UI.

confirm that BrainState is updated deterministically in some simple way.

Test 3: proactive interaction

Simulate a state that triggers ChatInterfaceModule.tick to produce speech.

Confirm that speech is produced without any external input, and that messages are routed correctly.