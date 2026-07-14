flowchart TD

subgraph group_swift["HAKI.app"]
  node_swift_app_shell["App Shell<br/>swift entrypoint<br/>[main.swift]"]
  node_swift_app_delegate["Delegate<br/>app lifecycle<br/>[AppDelegate.swift]"]
  node_core_process_manager["Core Manager<br/>process supervisor"]
  node_menu_bar_ui["Menu Bar<br/>ui shell<br/>[MenuBarUI.swift]"]
  node_permission_manager["Permissions<br/>security gate"]
  node_audio_stack["Voice Stack<br/>audio pipeline<br/>[AudioEngine.swift]"]
  node_capture_stack["Capture<br/>[ScreenReader.swift]"]
  node_os_actions["OS Actions<br/>system automation"]
  node_swift_store[("Local Store<br/>persistent storage<br/>[AppStore.swift]")]
  node_swift_ipc["IPC Client<br/>transport client<br/>[IPCClient.swift]"]
end

subgraph group_python["haki_core_service"]
  node_core_service["Service Main<br/>python entrypoint"]
  node_ipc_server["IPC Server<br/>grpc server<br/>[server.py]"]
  node_orchestrator["Orchestrator<br/>turn manager<br/>[orchestrator.py]"]
  node_intent_router["Intent Router<br/>routing<br/>[intent_router.py]"]
  node_dialogue_manager["Dialogue<br/>clarification"]
  node_planner["Planner<br/>planning<br/>[planner.py]"]
  node_safety_gate{{"Safety Gate<br/>policy check<br/>[safety_gate.py]"}}
  node_execution_engine["Executor<br/>action runner"]
  node_memory_brain[("Memory<br/>rag subsystem<br/>[memory_brain.py]")]
  node_learning_engine["Learning<br/>knowledge extraction<br/>[learning_engine.py]"]
  node_model_provider["Model Provider<br/>backend adapter<br/>[model_provider.py]"]
  node_language_engine["Language<br/>response shaping<br/>[language_engine.py]"]
  node_persona_engine["Persona<br/>style engine<br/>[persona_engine.py]"]
  node_mood_detector["Mood<br/>context signal<br/>[mood_detector.py]"]
  node_clock["Clock<br/>context signal<br/>[clock.py]"]
end

subgraph group_ipc["IPC contract"]
  node_ipc_proto["IPC Proto<br/>gRPC schema<br/>[haki_ipc.proto]"]
end

node_swift_app_shell -->|"boots"| node_swift_app_delegate
node_swift_app_delegate -->|"starts core"| node_core_process_manager
node_swift_app_delegate -->|"wires UI"| node_menu_bar_ui
node_swift_app_delegate -->|"checks access"| node_permission_manager
node_permission_manager -->|"unlocks"| node_audio_stack
node_permission_manager -->|"unlocks"| node_capture_stack
node_swift_store -->|"state"| node_swift_app_delegate
node_audio_stack -->|"sends audio"| node_swift_ipc
node_capture_stack -->|"sends context"| node_swift_ipc
node_os_actions -->|"reports effects"| node_swift_ipc
node_menu_bar_ui -->|"user turn"| node_swift_ipc
node_swift_ipc -->|"uses schema"| node_ipc_proto
node_ipc_server -->|"implements schema"| node_ipc_proto
node_core_process_manager -->|"launches"| node_core_service
node_core_service -->|"serves"| node_ipc_server
node_ipc_server -->|"dispatches"| node_orchestrator
node_orchestrator -->|"routes"| node_intent_router
node_orchestrator -->|"clarifies"| node_dialogue_manager
node_intent_router -->|"plans"| node_planner
node_planner -->|"validates"| node_safety_gate
node_safety_gate -->|"approves"| node_execution_engine
node_orchestrator -->|"retrieves"| node_memory_brain
node_learning_engine -->|"stores"| node_memory_brain
node_orchestrator -->|"learns"| node_learning_engine
node_orchestrator -->|"invokes models"| node_model_provider
node_dialogue_manager -->|"shapes reply"| node_language_engine
node_orchestrator -->|"styles voice"| node_persona_engine
node_orchestrator -->|"reads mood"| node_mood_detector
node_orchestrator -->|"reads time"| node_clock
node_execution_engine -->|"returns actions"| node_swift_ipc

click node_swift_app_shell "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/HAKI/main.swift"
click node_swift_app_delegate "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/HAKI/AppDelegate.swift"
click node_core_process_manager "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/HAKI/CoreProcessManager.swift"
click node_menu_bar_ui "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/UI/MenuBarUI.swift"
click node_permission_manager "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/Permissions/PermissionManager.swift"
click node_audio_stack "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/Audio/AudioEngine.swift"
click node_capture_stack "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/Capture/ScreenReader.swift"
click node_os_actions "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/OSActions/AppleScriptBridge.swift"
click node_swift_store "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/Store/AppStore.swift"
click node_swift_ipc "https://github.com/itwhizharsh/haki/blob/main/HAKI/Sources/Subsystems/IPC/IPCClient.swift"
click node_ipc_proto "https://github.com/itwhizharsh/haki/blob/main/proto/haki_ipc.proto"
click node_core_service "https://github.com/itwhizharsh/haki/blob/main/Core/haki_core_service.py"
click node_ipc_server "https://github.com/itwhizharsh/haki/blob/main/Core/core/ipc/server.py"
click node_orchestrator "https://github.com/itwhizharsh/haki/blob/main/Core/core/orchestrator/orchestrator.py"
click node_intent_router "https://github.com/itwhizharsh/haki/blob/main/Core/core/orchestrator/intent_router.py"
click node_dialogue_manager "https://github.com/itwhizharsh/haki/blob/main/Core/core/dialogue/dialogue_manager.py"
click node_planner "https://github.com/itwhizharsh/haki/blob/main/Core/core/planner/planner.py"
click node_safety_gate "https://github.com/itwhizharsh/haki/blob/main/Core/core/execution/safety_gate.py"
click node_execution_engine "https://github.com/itwhizharsh/haki/blob/main/Core/core/execution/execution_engine.py"
click node_memory_brain "https://github.com/itwhizharsh/haki/blob/main/Core/core/memory/memory_brain.py"
click node_learning_engine "https://github.com/itwhizharsh/haki/blob/main/Core/core/learning/learning_engine.py"
click node_model_provider "https://github.com/itwhizharsh/haki/blob/main/Core/core/model_provider/model_provider.py"
click node_language_engine "https://github.com/itwhizharsh/haki/blob/main/Core/core/language/language_engine.py"
click node_persona_engine "https://github.com/itwhizharsh/haki/blob/main/Core/core/persona/persona_engine.py"
click node_mood_detector "https://github.com/itwhizharsh/haki/blob/main/Core/core/mood/mood_detector.py"
click node_clock "https://github.com/itwhizharsh/haki/blob/main/Core/core/clock/clock.py"

classDef toneNeutral fill:#f8fafc,stroke:#334155,stroke-width:1.5px,color:#0f172a
classDef toneBlue fill:#dbeafe,stroke:#2563eb,stroke-width:1.5px,color:#172554
classDef toneAmber fill:#fef3c7,stroke:#d97706,stroke-width:1.5px,color:#78350f
classDef toneMint fill:#dcfce7,stroke:#16a34a,stroke-width:1.5px,color:#14532d
classDef toneRose fill:#ffe4e6,stroke:#e11d48,stroke-width:1.5px,color:#881337
classDef toneIndigo fill:#e0e7ff,stroke:#4f46e5,stroke-width:1.5px,color:#312e81
classDef toneTeal fill:#ccfbf1,stroke:#0f766e,stroke-width:1.5px,color:#134e4a
class node_swift_app_shell,node_swift_app_delegate,node_core_process_manager,node_menu_bar_ui,node_permission_manager,node_audio_stack,node_capture_stack,node_os_actions,node_swift_store,node_swift_ipc toneBlue
class node_core_service,node_ipc_server,node_orchestrator,node_intent_router,node_dialogue_manager,node_planner,node_safety_gate,node_execution_engine,node_memory_brain,node_learning_engine,node_model_provider,node_language_engine,node_persona_engine,node_mood_detector,node_clock toneAmber
class node_ipc_proto toneMintˍ