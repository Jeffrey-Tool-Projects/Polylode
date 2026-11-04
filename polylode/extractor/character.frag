// Forward preview adaptations of the source programs. Endfield's diffuse,
// ambient gradient and GGX terms: gl/shaders_endfield.py efMain/efDiffuse.
// GF2's ramp: uber PS ddbe031de613e4a704f89202824a9a3221dabd4325f271dc9a460d4ccb671614.
SHARED_VARS { vec4 finalColor; };
vec3 linearize(vec3 c) { return mix(c/12.92,pow((c+.055)/1.055,vec3(2.4)),step(vec3(.04045),c)); }
vec4 rampSample(vec2 uv) {
    vec4 value=textureLod(rampMap,uv,0.);
    if(rampSrgb) value.rgb=linearize(value.rgb);
    return value;
}
vec4 surfaceSample(vec2 uv) {
    vec4 value=texture(surfaceMap,uv);
    if(surfaceSrgb) value.rgb=linearize(value.rgb);
    return value;
}
vec3 saturateColor(vec3 c, float saturation) {
    return mix(vec3(dot(c, vec3(.212673, .715152, .072175))), c, saturation);
}
float luma(vec3 c) { return dot(c, vec3(.212673, .715152, .072175)); }
float efStep(float x) { x=clamp((x-.25)*1.333333,0.,1.);return x*x*(3.-2.*x); }
vec3 shadowLut(vec3 linearColor) {
    vec3 c=clamp(mix(linearColor*12.92,1.055*pow(max(linearColor,vec3(0)),vec3(.416667))-.055,
                    step(vec3(.0031308),linearColor)),0.,1.);
    float slice=floor(c.b*31.);
    vec2 uv=vec2(slice/32.+c.r*31./1024.+.5/1024.,c.g*31./32.+.5/32.);
    vec3 lo=textureLod(shadowLutMap,uv,0.).rgb, hi=textureLod(shadowLutMap,uv+vec2(1./32.,0),0.).rgb;
    if(shadowLutSrgb) {lo=linearize(lo);hi=linearize(hi);}
    return mix(lo,hi,c.b*31.-slice);
}
void MAIN() {
    vec4 base = tint;
    if (hasBase) {
        vec2 uv=UV0*baseUv.xy;
        float cs=cos(baseRotation), sn=sin(baseRotation);
        uv=mat2(cs,sn,-sn,cs)*uv+baseUv.zw;
        vec4 sampled = texture(baseMap, uv);
        vec3 linearColor=linearize(sampled.rgb);
        base *= vec4(linearColor,sampled.a);
    }
    if(base.a < cutoff) discard;
    vec3 N=normalize(VAR_WORLD_NORMAL);
    if(!gl_FrontFacing) N=-N;
    vec3 L=normalize(vec3(.147, .530, .835));
    vec3 V=normalize(CAMERA_POSITION-VAR_WORLD_POSITION);
    float nl=dot(N,L), nv=max(dot(N,V),0.);
    vec3 color;
    if(game==1) {
        vec4 mgm=hasSurface ? surfaceSample(UV0) : vec4(metallic,specularLevel,1,smoothness);
        vec3 cameraForward=normalize(CAMERA_DIRECTION);
        float ndl=clamp(nl-.4,-1.,1.);
        vec4 ramp=hasRamp ? rampSample(vec2(.5+.5*ndl,.5)) : vec4(1,1,1,efStep(ndl));
        float viewDot=dot(N,cameraForward);
        float viewA=hasRamp ? rampSample(vec2(.5+.5*viewDot,.5)).a : efStep(viewDot);
        float kd=.96*(1.-mgm.r);
        vec3 albedo=base.rgb*kd;
        vec3 shadow=(shadowLutAvailable ? shadowLut(base.rgb) : saturateColor(base.rgb*shadowBrightness,shadowSaturation))*kd*.8;
        float lit=min(ramp.a,mgm.b);
        vec3 deep=saturateColor(shadow*.65,1.2);
        vec3 c1=mix(mix(deep,shadow,clamp(viewA*mgm.b+ramp.a,0.,1.)),albedo,lit);
        float chroma=max(ramp.r,max(ramp.g,ramp.b))-min(ramp.r,min(ramp.g,ramp.b));
        vec3 c2=c1*mix(vec3(1),ramp.rgb,chroma);
        vec3 diffuse=c2*clamp(luma(c1)/max(luma(c2),.001),0.,1.5);
        float grad=clamp(N.y+.15,0.,1.)*1.5+.5;
        vec3 lighting=(mix(vec3(.7830188,.8293082,1),vec3(1),lit)*grad+vec3(.9))*.45;
        vec3 H=normalize(L+2.*normalize(vec3(cameraForward.x,L.y,cameraForward.z))+3.*V);
        float alpha=max((1.-mgm.a)*(1.-mgm.a),.0078125);
        float a2=alpha*alpha, nh=dot(N,H);
        float d=(nh*a2-nh)*nh+1.;
        float ggx=clamp(a2/max(d*d,1e-12)*.5/(2.*nv+alpha+.0001)-.000061,0.,20.);
        vec3 f0=mix(vec3(.04*mgm.g),base.rgb,mgm.r);
        color=diffuse*lighting+f0*ggx*lighting*(.5+.5*lit)*mix(.8,1.,lit);
        float k=clamp(luma(color)-.5,0.,.5);
        color=mix(vec3(luma(color)),color,1.+k*k);
    } else if(game==2) {
        vec4 rmo=hasSurface ? surfaceSample(UV0) : vec4(1.-smoothness,metallic,1,0);
        float rough=max(rmo.r,.001), metal=rmo.g;
        vec3 f0=mix(vec3(.04),base.rgb,metal);
        float lambert=max(nl,0.);
        float x=max(lambert*(3.*.333299994468689),.00006103515625);
        vec3 ramp=hasRamp ? rampSample(vec2(x,.125)).rgb : vec3(lambert);
        if(hasRamp && x>.00006103515625) ramp*=min(lambert/x,1.);
        vec3 H=normalize(V+L);
        float nh=max(dot(N,H),0.), vh=max(dot(V,H),0.), lh=max(dot(L,H),0.);
        nv=min(nv+.00001,1.);
        float a=rough*rough;
        float D=min(pow(a/(nh*nh*a*a+1.-nh*nh),2.),2048.);
        float vis=min(.5/max(lambert*(nv*(1.-a)+a)+nv*(lambert*(1.-a)+a),.0001),1.);
        float fres=1.+(clamp(50.*f0.g,0.,1.)-1.)*pow(max(1.-vh,.001),5.);
        vec3 spec=vec3(D*vis*fres);
        if(hasRamp) {
            float peakD=min(pow(a/(a*a),2.),2048.);
            float peakV=min(.5/max(vh*(lh*(1.-a)+a)+lh*(vh*(1.-a)+a),.0001),1.);
            spec=peakD*peakV*fres*rampSample(vec2(clamp(D*vis/(peakD*peakV),0.,1.),.375)).rgb;
        }
        color=(base.rgb*(1.-metal)+f0*clamp(spec,0.,10.))*ramp+base.rgb*emission*rmo.a;
    } else {
        // M_Char_Com 295CCDA... base pass, lines 196–237; KuroToon
        // F42E... diffuse, lines 353–434. Screen shadows/AO neutralized.
        float viewLight=.5*(dot(L,V)*(1.-abs(V.y))+1.);
        float threshold=(1.-clamp(dot(N,V),0.,1.))*mix(backShadowOffset,frontShadowOffset,viewLight)+shadowProcess;
        float q=smoothstep(threshold-shadowWidth,threshold+shadowWidth,.5*nl+.5);
        float upperBand=q*q*(3.-2.*q);
        float separation=min(10.*dot(abs(sceneLight-sceneShadow),vec3(.3,.59,.11)),1.);
        separation=separation*separation*(3.-2.*separation);
        float u=pow(clamp(1.-smoothness,0.,1.),.1);
        float exponent=1000.-999.*u, scale=.1+19.9*(1.-u);
        bool metal=metallic>=.85;
        float power=metal ? metalSpecularPower : specularPower;
        vec3 mirrored=L;
        if(q<.4 && !metal) mirrored.xz=-mirrored.xz;
        float ndh=clamp(dot(N,normalize(V+mirrored)),0.,1.);
        float lobe=clamp((pow(ndh,exponent)*power*(power+2.)/(2.*3.14159265)-.6)*(4./3.),0.,1.);
        lobe=lobe*lobe*(3.-2.*lobe)*scale;
        vec3 specColor=mix(vec3(.04+.04*metallic),base.rgb,metallic);
        vec3 materialColor=base.rgb*(1.-metallic*pbrRate)+specColor*pbrRate*lobe;
        color=materialColor*mix(sceneShadow,sceneLight,upperBand*separation);
    }
    // the view's light from straight above, on what faces up, over each game's own lighting
    color+=base.rgb*topLight*.5*max(N.y,0.);
    BASE_COLOR=vec4(0,0,0,base.a);
    SHARED.finalColor=vec4(color*intensity,base.a);
}
void POST_PROCESS() { COLOR_SUM=SHARED.finalColor; }
