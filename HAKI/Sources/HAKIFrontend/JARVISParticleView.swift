// JARVISParticleView.swift
// HAKIFrontend — SceneKit 3D JARVIS HUD (NSViewRepresentable)
// Requirements: 5.1, 14.5

import SwiftUI
import SceneKit
import AppKit

// MARK: - JARVISParticleView

struct JARVISParticleView: NSViewRepresentable {
    @Binding var audioLevel: Float
    @Environment(HAKIStateModel.self) private var stateModel

    // MARK: - Coordinator factory

    func makeCoordinator() -> Coordinator {
        Coordinator()
    }

    // MARK: - NSViewRepresentable

    func makeNSView(context: Context) -> SCNView {
        let scnView = SCNView()
        scnView.antialiasingMode = .multisampling4X
        scnView.backgroundColor = .clear
        scnView.allowsCameraControl = false
        scnView.delegate = context.coordinator

        do {
            try context.coordinator.buildScene(in: scnView)
        } catch {
            print("[JARVISParticleView] Scene setup failed: \(error)")
            scnView.scene = SCNScene() // fallback to empty scene per Req 14.5
        }
        return scnView
    }

    func updateNSView(_ nsView: SCNView, context: Context) {
        // Push latest values into coordinator so per-frame renderer logic can read them
        context.coordinator.audioLevel = audioLevel
        context.coordinator.currentState = stateModel.currentState
    }

    // MARK: - Coordinator

    final class Coordinator: NSObject, SCNSceneRendererDelegate {

        // Per-frame snapshot properties updated from updateNSView(_:context:)
        var audioLevel: Float = 0.0
        var currentState: HAKIState = .idle

        // Scene object references populated by buildScene(in:)
        var particleSystem: SCNParticleSystem?
        var sphereNode: SCNNode?
        var ringNodes: [SCNNode] = []

        // State tracking for change detection in the render loop
        var previousState: HAKIState = .idle

        // MARK: - Per-frame render delegate (Task 6.3)
        // Requirements: 5.4, 5.6, 5.7, 5.9, 5.10, 5.11

        func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
            // Take a snapshot of coordinator properties — renderer runs on SceneKit's render thread.
            let level = audioLevel
            let state = currentState
            let prevState = previousState

            // MARK: Req 5.6 — birthRate = audioLevel * particleEmissionRate + particleEmissionRate
            if let ps = particleSystem {
                ps.birthRate = CGFloat(level * state.particleEmissionRate + state.particleEmissionRate)
            }

            // MARK: Req 5.7 — sphere scale driven by audio level (unless idle pulse is active)
            if let sphere = sphereNode, state != .idle {
                let s = 1.0 + Double(level) * 0.5
                sphere.scale = SCNVector3(s, s, s)
            }

            // MARK: State transition handling
            guard state != prevState else { return }
            previousState = state

            // MARK: Req 5.4 — update ring material emission colours on state change
            let accentNSColor = NSColor(state.accentColor)
            for ring in ringNodes {
                ring.geometry?.firstMaterial?.emission.contents = accentNSColor
            }

            // MARK: Ring rotation durations per state — Req 5.3 (idle/default), 5.9 (thinking)
            let rotationConfigs: [(x: CGFloat, y: CGFloat, z: CGFloat, duration: TimeInterval)]
            switch state {
            case .thinking:
                // Req 5.9: faster ring rotations
                rotationConfigs = [
                    (x: 1.0, y: 0.3, z: 0.0, duration: 1.5),
                    (x: 0.2, y: 1.0, z: 0.4, duration: 1.1),
                    (x: 0.5, y: 0.2, z: 1.0, duration: 2.0),
                ]
            default:
                // Default durations matching buildScene exactly
                rotationConfigs = [
                    (x: 1.0, y: 0.3, z: 0.0, duration: 6.0),
                    (x: 0.2, y: 1.0, z: 0.4, duration: 4.5),
                    (x: 0.5, y: 0.2, z: 1.0, duration: 8.0),
                ]
            }

            for (index, ring) in ringNodes.enumerated() {
                guard index < rotationConfigs.count else { continue }
                let cfg = rotationConfigs[index]
                ring.removeAllActions()
                ring.runAction(
                    SCNAction.repeatForever(
                        SCNAction.rotateBy(x: cfg.x, y: cfg.y, z: cfg.z, duration: cfg.duration)
                    )
                )
            }

            // MARK: Sphere idle pulse and error jitter
            if let sphere = sphereNode {
                switch state {
                case .idle:
                    // Req 5.8 — restore ambient pulse when returning to idle
                    sphere.removeAction(forKey: "errorJitter")
                    if sphere.action(forKey: "idlePulse") == nil {
                        let pulse = SCNAction.repeatForever(
                            SCNAction.sequence([
                                SCNAction.scale(to: 1.05, duration: 1.0),
                                SCNAction.scale(to: 0.95, duration: 1.0),
                            ])
                        )
                        sphere.runAction(pulse, forKey: "idlePulse")
                    }

                case .error:
                    // Req 5.10 — ±0.05 pt position jitter for ~1.0 s then return to origin
                    sphere.removeAction(forKey: "idlePulse")
                    sphere.removeAction(forKey: "errorJitter")
                    // Reset scale to neutral before jitter
                    sphere.scale = SCNVector3(1, 1, 1)

                    let right   = SCNAction.moveBy(x:  0.05, y: 0, z: 0, duration: 0.1)
                    let left    = SCNAction.moveBy(x: -0.05, y: 0, z: 0, duration: 0.1)
                    let right2  = SCNAction.moveBy(x:  0.05, y: 0, z: 0, duration: 0.1)
                    let left2   = SCNAction.moveBy(x: -0.05, y: 0, z: 0, duration: 0.1)
                    let right3  = SCNAction.moveBy(x:  0.05, y: 0, z: 0, duration: 0.1)
                    let left3   = SCNAction.moveBy(x: -0.05, y: 0, z: 0, duration: 0.1)
                    let right4  = SCNAction.moveBy(x:  0.05, y: 0, z: 0, duration: 0.1)
                    let left4   = SCNAction.moveBy(x: -0.05, y: 0, z: 0, duration: 0.1)
                    let right5  = SCNAction.moveBy(x:  0.05, y: 0, z: 0, duration: 0.1)
                    let left5   = SCNAction.moveBy(x: -0.05, y: 0, z: 0, duration: 0.1)
                    // Return to exact origin after 10 × 0.1 s = 1.0 s total
                    let returnToOrigin = SCNAction.move(to: SCNVector3Zero, duration: 0)
                    let jitter = SCNAction.sequence([
                        right, left, right2, left2, right3, left3, right4, left4, right5, left5,
                        returnToOrigin,
                    ])
                    sphere.runAction(jitter, forKey: "errorJitter")

                default:
                    // Leaving idle — remove pulse so per-frame scale takes over (Req 5.7)
                    sphere.removeAction(forKey: "idlePulse")
                    sphere.removeAction(forKey: "errorJitter")
                }
            }
        }

        // MARK: - Scene construction (Task 6.2)
        // Requirements: 5.2, 5.3, 5.4, 5.5, 5.8
        func buildScene(in scnView: SCNView) throws {
            let scene = SCNScene()
            scnView.scene = scene

            // MARK: Camera — Req 5.1
            let camera = SCNCamera()
            let cameraNode = SCNNode()
            cameraNode.camera = camera
            cameraNode.position = SCNVector3(0, 0, 5)
            scene.rootNode.addChildNode(cameraNode)

            // MARK: Three torus ring nodes — Req 5.2, 5.3, 5.4
            // (ringRadius, duration, rotateBy axes)
            let ringConfigs: [(ringRadius: CGFloat, duration: TimeInterval, x: CGFloat, y: CGFloat, z: CGFloat)] = [
                (1.0, 6.0,  1.0, 0.3, 0.0),
                (1.4, 4.5,  0.2, 1.0, 0.4),
                (1.8, 8.0,  0.5, 0.2, 1.0),
            ]

            ringNodes = []
            for config in ringConfigs {
                // Geometry — Req 5.2
                let torus = SCNTorus()
                torus.ringRadius = config.ringRadius
                torus.pipeRadius = 0.04

                // Material emission set to current accent colour — Req 5.4
                let material = SCNMaterial()
                material.emission.contents = NSColor(currentState.accentColor)
                torus.materials = [material]

                // Wrap in node
                let ringNode = SCNNode(geometry: torus)

                // Permanently running rotation action — Req 5.3
                let rotateAction = SCNAction.repeatForever(
                    SCNAction.rotateBy(
                        x: config.x,
                        y: config.y,
                        z: config.z,
                        duration: config.duration
                    )
                )
                ringNode.runAction(rotateAction)

                scene.rootNode.addChildNode(ringNode)
                ringNodes.append(ringNode)
            }

            // MARK: Central sphere with particle system — Req 5.5
            let sphere = SCNSphere(radius: 0.3)
            let sphereMaterial = SCNMaterial()
            sphereMaterial.emission.contents = NSColor(currentState.accentColor)
            sphere.materials = [sphereMaterial]

            let newSphereNode = SCNNode(geometry: sphere)
            newSphereNode.position = SCNVector3(0, 0, 0)

            // Particle system — Req 5.5
            let ps = SCNParticleSystem()
            ps.particleSize = 0.05
            ps.particleColor = NSColor(currentState.accentColor)
            ps.emissionDuration = 0          // continuous emission
            ps.birthRate = 20               // matches idle particleEmissionRate baseline
            newSphereNode.addParticleSystem(ps)

            scene.rootNode.addChildNode(newSphereNode)

            // Store references for per-frame updates (task 6.3)
            sphereNode = newSphereNode
            particleSystem = ps

            // MARK: Idle ambient pulse — Req 5.8
            // Scale sphere 0.95 ↔ 1.05 over 2s, repeat forever
            if currentState == .idle {
                let scaleUp   = SCNAction.scale(to: 1.05, duration: 1.0)
                let scaleDown = SCNAction.scale(to: 0.95, duration: 1.0)
                let pulse = SCNAction.repeatForever(SCNAction.sequence([scaleUp, scaleDown]))
                newSphereNode.runAction(pulse, forKey: "idlePulse")
            }
        }
    }
}
