'use client';

import { useRef, useState, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Stars, useTexture } from '@react-three/drei';
import * as THREE from 'three';

interface CityMarkerProps {
  coordinates: [number, number];
  onClick: () => void;
}

function CityMarker({ coordinates, onClick }: CityMarkerProps) {
  const [hovered, setHovered] = useState(false);
  const meshRef = useRef<THREE.Mesh>(null);
  
  // 将经纬度转换为球面坐标
  const position = useMemo(() => {
    const [lon, lat] = coordinates;
    const radius = 1.02;
    const phi = (90 - lat) * (Math.PI / 180);
    const theta = (lon + 180) * (Math.PI / 180);
    
    return new THREE.Vector3(
      -radius * Math.sin(phi) * Math.cos(theta),
      radius * Math.cos(phi),
      radius * Math.sin(phi) * Math.sin(theta)
    );
  }, [coordinates]);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.scale.setScalar(hovered ? 1.5 : 1);
    }
  });

  return (
    <mesh
      ref={meshRef}
      position={position}
      onClick={onClick}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[0.02, 16, 16]} />
      <meshStandardMaterial
        color={hovered ? '#ff6b9d' : '#00d4ff'}
        emissive={hovered ? '#ff6b9d' : '#00d4ff'}
        emissiveIntensity={hovered ? 0.8 : 0.5}
      />
    </mesh>
  );
}

interface EarthProps {
  onCityClick: (cityId: string) => void;
  cityCoordinates: [number, number][];
}

function Earth({ onCityClick, cityCoordinates }: EarthProps) {
  const earthRef = useRef<THREE.Mesh>(null);
  
  const [earthTexture, bumpTexture] = useTexture([
    'https://unpkg.com/three-globe/example/img/earth-blue-marble.jpg',
    'https://unpkg.com/three-globe/example/img/earth-topology.png',
  ]);

  useFrame(() => {
    if (earthRef.current) {
      earthRef.current.rotation.y += 0.0005;
    }
  });

  return (
    <group>
      {/* 地球 */}
      <mesh ref={earthRef}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshStandardMaterial
          map={earthTexture}
          bumpMap={bumpTexture}
          bumpScale={0.02}
          roughness={0.8}
          metalness={0.1}
        />
      </mesh>
      
      {/* 城市标记点 */}
      {cityCoordinates.map((coords, index) => (
        <CityMarker
          key={index}
          coordinates={coords}
          onClick={() => onCityClick(`city-${index}`)}
        />
      ))}
      
      {/* 大气层光晕 */}
      <mesh scale={[1.02, 1.02, 1.02]}>
        <sphereGeometry args={[1, 64, 64]} />
        <meshBasicMaterial
          color="#4a90d9"
          transparent
          opacity={0.1}
          side={THREE.BackSide}
        />
      </mesh>
    </group>
  );
}

interface GlobeCanvasProps {
  onCityClick: (cityId: string) => void;
  cityCoordinates: [number, number][];
}

export function GlobeCanvas({ onCityClick, cityCoordinates }: GlobeCanvasProps) {
  return (
    <Canvas camera={{ position: [0, 0, 2.2], fov: 45 }}>
      <ambientLight intensity={0.6} />
      <pointLight position={[10, 10, 10]} intensity={1.2} />
      <pointLight position={[-10, -10, -10]} intensity={0.3} color="#9d4edd" />
      
      <Stars
        radius={100}
        depth={50}
        count={3000}
        factor={4}
        saturation={0}
        fade
        speed={0.5}
      />
      
      <Earth onCityClick={onCityClick} cityCoordinates={cityCoordinates} />
      
      <OrbitControls
        enableZoom={true}
        enablePan={false}
        minDistance={1.3}
        maxDistance={3.5}
        autoRotate
        autoRotateSpeed={0.3}
        dampingFactor={0.1}
        enableDamping
      />
    </Canvas>
  );
}
